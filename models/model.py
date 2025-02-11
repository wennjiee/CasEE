from models.layers import *


# 事件类型抽取
class TypeCls(nn.Module):
    def __init__(self, config):
        super(TypeCls, self).__init__()
        self.type_emb = nn.Embedding(config.type_num, config.hidden_size)  # 10*768
        self.register_buffer('type_indices', torch.arange(0, config.type_num, 1).long())
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.config = config
        self.Predictor = AdaptiveAdditionPredictor(config.hidden_size, dropout_rate=config.decoder_dropout)

    def forward(self, text_rep, mask):
        type_emb = self.type_emb(self.type_indices)  # randomly init type embedding, 10*768 = 10*768(1*10)
        pred = self.Predictor(type_emb, text_rep, mask)  # [b, c] batch_size*10的概率？
        p_type = torch.sigmoid(pred)  # 1*10 c_hat
        return p_type, type_emb


# 触发词抽取
class TriggerRec(nn.Module):
    def __init__(self, config, hidden_size):
        super(TriggerRec, self).__init__()

        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)

        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)

        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.head_cls = nn.Linear(hidden_size, 1, bias=True)
        self.tail_cls = nn.Linear(hidden_size, 1, bias=True)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.config = config

    def forward(self, query_emb, text_emb, mask):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''
        # trigger_rec(type_rep, output_emb, mask) self.type_emb(self.type_indices = type_emb = query_emb, text_emb是句嵌入
        h_cln = self.ConditionIntegrator(text_emb, query_emb)  # hi c
        h_cln = self.dropout(h_cln)  # b*400*768

        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)  # 2*400*768

        inp = self.layer_norm(h_sa + h_cln)  # LayerNorm归一化的一种方法
        inp = gelu(self.hidden(inp))  # 激活函数 gelu(x) = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) 高斯误差
        inp = self.dropout(inp)  # b*400*768

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1] self.head/tail_cls = nn.Linear(int=768, out=1,bias=true)
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]

        return p_s, p_e, h_cln


# 论元抽取
class ArgsRec(nn.Module):
    def __init__(self, config, hidden_size, num_labels, seq_len, pos_emb_size):
        super(ArgsRec, self).__init__()
        self.relative_pos_embed = nn.Embedding(seq_len * 2, pos_emb_size)
        self.ConditionIntegrator = ConditionalLayerNorm(hidden_size)
        self.SA = MultiHeadedAttention(hidden_size, heads_num=config.decoder_num_head, dropout=config.decoder_dropout)
        self.hidden = nn.Linear(hidden_size + pos_emb_size, hidden_size)

        self.head_cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.tail_cls = nn.Linear(hidden_size, num_labels, bias=True)

        self.gate_hidden = nn.Linear(hidden_size, hidden_size)
        self.gate_linear = nn.Linear(hidden_size, num_labels)

        self.seq_len = seq_len
        self.dropout = nn.Dropout(config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.config = config

        # self.glocal_att_tri = GLFusion()
        # self.global_bigram_emb = torch.load('global_bigram_emb.pth', map_location="cuda:0")
        # self.trigger_proj = nn.Linear(768,  768)
        # self.weight =  nn.Parameter(torch.ones(1))

    # self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
    def forward(self, text_emb, relative_pos, trigger_mask, mask, type_emb):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t, e]
        :param relative_pos: [b, t, e]
        :param trigger_mask: [b, t]
        :param mask:
        :param type_emb: [b, e]
        :return:  [b, t, a], []
        '''
        # text_rep_type = h_cln = self.ConditionIntegrator(text_emb, query_emb) # hi c
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2  # b*768 平均值作为表征触发词

        h_cln = self.ConditionIntegrator(text_emb, trigger_emb)  # integrate the trigger info. into text_emb 源代码
        h_cln = self.dropout(h_cln)
        # glocal_tri = self.glocal_att_tri(text_emb, self.global_bigram_emb, trigger_emb, mask)
        # h_cln = self.ConditionIntegrator(glocal_tri, trigger_emb)

        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        h_sa = self.layer_norm(h_sa + h_cln)  # LayerNorm

        rp_emb = self.relative_pos_embed(relative_pos)  # nn.Embedding(800,64)
        rp_emb = self.dropout(rp_emb)

        inp = torch.cat([h_sa, rp_emb], dim=-1)  # Zct = [Zct';P] 拼接距离位置特征
        inp = gelu(self.hidden(inp))  # self.hidden = nn.Linear(int=832,out=768,bias=true)
        inp = self.dropout(inp)

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l] self.head/tail_cls = nn.Linear(int=768, out=18,bias=true)
        p_e = torch.sigmoid(self.tail_cls(inp))

        # I(r,c) = type_soft_constrain
        # [b, l] self.gate_linear = nn.Linear(int=768, out=18,bias=true)
        type_soft_constrain = torch.sigmoid(self.gate_linear(type_emb))
        # type_soft_constrain[0][1] = type_soft_constrain[0][2 - 400] = 上述结果
        type_soft_constrain = type_soft_constrain.unsqueeze(1).expand_as(p_s)  # b*400*18

        p_s = p_s * type_soft_constrain
        p_e = p_e * type_soft_constrain

        return p_s, p_e, type_soft_constrain


class CasEE(nn.Module):
    def __init__(self, config, model_weight, pos_emb_size):
        super(CasEE, self).__init__()

        self.bert = model_weight
        self.config = config
        self.args_num = config.args_num
        self.text_seq_len = config.seq_length

        self.type_cls = TypeCls(config)
        self.trigger_rec = TriggerRec(config, config.hidden_size)
        self.args_rec = ArgsRec(config, config.hidden_size, self.args_num, self.text_seq_len, pos_emb_size)

        self.glocal_att = GLFusion()
        # self.cross_att = MultiHeadedAttention(768, heads_num=1, dropout=0.3)
        # self.cross_proj = nn.Linear(768 * 2, 768)

        self.dropout = nn.Dropout(config.decoder_dropout)
        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))
        self.weight3 = nn.Parameter(torch.ones(1))

    def get_bigram_emb(self, tokens, mask, trigger_mask, segment):
        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]
        # 用 + 的方式取代词的表示 b*768 = b*768[c1] + b*768[c2]
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), output_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2  # b*768 取平均
        return trigger_emb

    def forward(self, tokens, segment, mask, type_id, type_vec, trigger_s_vec, trigger_e_vec, relative_pos,
                trigger_mask, args_s_vec, args_e_vec, args_mask, tokens_syntac_ids):
        '''
        :self.model(token, seg, mask, d_t, t_v, t_s, t_e, r_pos, t_m, a_s, a_e, a_m)
        :param tokens: [b, t]
        :param segment: [b, t]
        :param mask: [b, t], 0 if masked
        :param trigger_s: [b, t]
        :param trigger_e: [b, t]
        :param relative_pos:
        :param trigger_mask: [0000011000000]
        :param args_s: [b, l, t]
        :param args_e: [b, l, t]
        :param args_m: [b, k]
        :return:
        '''

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]  # batch_size*400*768

        # outputs_syn = self.bert(
        #     tokens_syntac_ids,
        #     attention_mask=mask,
        #     token_type_ids=segment,
        #     position_ids=None,
        #     head_mask=None,
        #     inputs_embeds=None,
        #     output_attentions=None,
        #     output_hidden_states=None,
        # )
        # output_syn_emb = outputs_syn[0]

        # type_cls 输出预测矩阵p_type, 随机初始化的type_emb
        p_type, type_emb = self.type_cls(output_emb, mask)  # batch_size*10, 10*768
        p_type = p_type.pow(self.config.pow_0)
        type_loss = self.loss_0(p_type, type_vec)  # 预测与真实loss
        type_loss = torch.sum(type_loss)
        type_rep = type_emb[type_id, :]  # batch_size*768 随机初始化emb中type_id事件类型id项

        # cross_att = output_syn_emb + output_emb
        # cross_att = self.cross_att(output_emb, output_emb, output_syn_emb, mask)
        e_global = self.glocal_att(output_emb, type_emb, type_rep, mask)

        # trigger_rec p_s=p_e = 1*400*1, text_rep_type=batch_size*400*768
        # text_rep_type = h_cln = ConditionIntegrator(text_emb, query_emb) 条件注入后的表示
        # p_s, p_e, text_rep_type = self.trigger_rec(type_rep, output_emb, mask) # 源代码
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, e_global, mask)
        p_s = p_s.pow(self.config.pow_1)  # self.config.pow_1 = 1
        p_e = p_e.pow(self.config.pow_1)
        p_s = p_s.squeeze(-1)
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.loss_1(p_s, trigger_s_vec)  # 预测trigger start位置 与 真实start位置误差
        trigger_loss_e = self.loss_1(p_e, trigger_e_vec)  # 预测trigger end位置   与 真实end位置误差
        mask_t = mask.float()  # [b, t]
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))

        # args_rec trigger_mask = mask = b*400
        # text_rep_type = h_cln是全局与局部条件注入后的特征表示
        # 源代码如下
        # p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        # cross_att = text_rep_type + output_syn_emb
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        p_s = p_s.pow(self.config.pow_2)  # self.config.pow_2 = 1
        p_e = p_e.pow(self.config.pow_2)
        args_loss_s = self.loss_2(p_s, args_s_vec.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.loss_2(p_e, args_e_vec.transpose(1, 2))
        mask_a = mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = torch.sum(args_loss_s.mul(mask_a))
        args_loss_e = torch.sum(args_loss_e.mul(mask_a))

        trigger_loss = trigger_loss_s + trigger_loss_e
        args_loss = args_loss_s + args_loss_e

        type_loss = self.config.w1 * type_loss
        trigger_loss = self.config.w2 * trigger_loss
        args_loss = self.config.w3 * args_loss
        loss = type_loss + trigger_loss + args_loss

        return loss, type_loss, trigger_loss, args_loss

    def plm(self, tokens, segment, mask):
        assert tokens.size(0) == 1

        outputs = self.bert(
            tokens,
            attention_mask=mask,
            token_type_ids=segment,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
        )
        output_emb = outputs[0]
        return output_emb

    def predict_type(self, text_emb, mask):
        assert text_emb.size(0) == 1
        p_type, type_emb = self.type_cls(text_emb, mask)
        p_type = p_type.view(self.config.type_num).data.cpu().numpy()
        return p_type, type_emb

    def predict_trigger(self, type_rep, text_emb, mask):
        assert text_emb.size(0) == 1
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, text_emb, mask)
        p_s = p_s.squeeze(-1)  # [b, t]
        p_e = p_e.squeeze(-1)
        mask = mask.float()  # [1, t]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.text_seq_len).data.cpu().numpy()
        return p_s, p_e, text_rep_type

    def predict_args(self, text_rep_type, relative_pos, trigger_mask, mask, type_rep):
        assert text_rep_type.size(0) == 1
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.text_seq_len, self.args_num).data.cpu().numpy()
        return p_s, p_e, type_soft_constrain


class GLFusion(nn.Module):
    def __init__(self):
        super(GLFusion, self).__init__()
        self.GSA = GlobalMHAttention(hidden_size=768, heads_num=1, dropout=0.3)
        self.GATE1 = GATE()
        self.GATE2 = GATE()

    def forward(self, h_text_rep, e_type_emb, type_rep, mask):
        gloalEAtt = self.GSA(h_text_rep, e_type_emb, e_type_emb, mask)  # 4*400*768
        Hg = self.GATE1(h_text_rep, gloalEAtt)
        et = type_rep.unsqueeze(1)
        et = et.repeat(1, 400, 1)
        Vt = self.GATE2(Hg, et)
        return Vt


class GATE(nn.Module):
    def __init__(self):
        super(GATE, self).__init__()
        self.gfunc = nn.Linear(768 * 2, 768)

    def forward(self, p, q):
        pq = torch.cat([p, q], dim=-1)
        g = self.gfunc(pq)
        g = torch.sigmoid(g)
        gate_res = g * p + (1 - g) * q
        return gate_res
