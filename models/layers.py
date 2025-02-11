from torch import nn
import math
import torch


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.eps = eps
        self.gamma_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.beta_dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

        nn.init.zeros_(self.gamma_dense.weight)
        nn.init.zeros_(self.beta_dense.weight)

    # 利用CLN来将外部条件融入到预训练模型中，其直接应用是条件文本生成
    # 在Bert等Transformer模型中，主要的Normalization方法是Layer Normalization，
    # 自然想到将对应的 β 和 γ 变成输入条件的函数，来控制 Transformer 模型的生成行为，

    def forward(self, x, condition):
        '''
        self.ConditionIntegrator(x=text_emb, condition=query_emb) # hi c
        :param x: [b, t, e]
        :param condition: [b, e]
        :return:
        '''
        mean = x.mean(-1, keepdim=True)  # b*400*1
        std = x.std(-1, keepdim=True)  # b*400*1

        condition = condition.unsqueeze(1).expand_as(x)  # b*400*768
        gamma = self.gamma_dense(
            condition) + self.gamma  # gain nn.Linear(in_features=768,out=768,bias=false) + nn.Parameter(torch.ones(hidden_size))
        beta = self.beta_dense(
            condition) + self.beta  # bias nn.Linear(in_features=768,out=768,bias=false) + nn.Parameter(torch.zeros(hidden_size))
        x = gamma * (x - mean) / (std + self.eps) + beta
        return x


class AdaptiveAdditionPredictor(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.0):
        super(AdaptiveAdditionPredictor, self).__init__()
        self.v = nn.Linear(hidden_size * 4, 1)
        self.hidden = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.dropout = nn.Dropout(dropout_rate)

        self.U1 = nn.Linear(hidden_size, hidden_size * 4)
        self.d_ = math.sqrt(hidden_size)
        self.U2 = nn.Linear(hidden_size, hidden_size * 4)

    # c hi mask query代表10个事件类别
    def forward(self, query, context, mask):
        '''
        :param query: [c, e]
        :param context: [b, t, e]
        :param mask: [b, t], 0 if masked
        :return: [b, e]
        '''
        # b*10*400*768 先降维再升维 b*400*768
        # hi and c query代表randomly init type embedding, 10*768 = 10*768(1*10), context代表原文的embedding 2*10*400*768
        # context_[0][0] = context_[0][1-9]

        context_ = context.unsqueeze(1).expand(context.size(0), query.size(0), context.size(1), context.size(2))  # [b, c, t, e]

        # query_[0][0][0] = query_[0][0][399] 表示第0个类型的embedding
        query_ = query.unsqueeze(0).unsqueeze(2).expand_as(context_)  # [b, c, t, e] 2*10*400*768

        # c=query_与hi=context_之间的相似度 v`*tanh(W[c;hi,|c-hi|;c.*hi]) 双曲正切 c是事件类型 hi是context_
        xi = torch.cat([query_, context_, torch.abs(query_ - context_), query_ * context_ / self.d_], dim=-1)
        scores = self.v(torch.tanh(self.hidden(xi) + self.U1(query_)))  # [b, c, t, 1]
        scores = self.dropout(scores)  # b*10*400*1

        mask = (mask < 1).unsqueeze(1).unsqueeze(3).expand_as(scores)  # [b, c, t, 1]
        scores = scores.masked_fill_(mask, -1e10)
        scores = scores.transpose(-1, -2)  # [b, c, 1, t]
        scores = torch.softmax(scores, dim=-1)  # [b, c, 1, t] 2*10*1*400 位置加权
        g = torch.matmul(scores, context_).squeeze(2)  # [b, c, e] 论文中Sc

        query = query.unsqueeze(0).expand_as(g)  # [b, c, e] 论文中c
        # 双曲正切激活函数 计算Sc与c相似程度
        pred = self.v(torch.tanh(
            self.hidden(torch.cat([query, g, torch.abs(query - g), query * g / self.d_], dim=-1)) + self.U2(
                query))).squeeze(-1)  # [b, c]

        return pred


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size  x seq_length]
            mask is 0 if it is masked

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()  # 2*400*768
        heads_num = self.heads_num  # 1
        per_head_size = self.per_head_size  # 768

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        # 输出均为batch_size*1*400*768
        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2) for l, x in
                             zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))

        mask = mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        scores = scores + mask

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)  # nn.Linear(768.768)

        return output


class GlobalMHAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(GlobalMHAttention, self).__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size  x seq_length]
            mask is 0 if it is masked

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = query.size()  # 2*400*768
        heads_num = self.heads_num  # 1
        per_head_size = self.per_head_size  # 768

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        # 输出均为batch_size*1*400*768        
        query_ = self.w_q(query)
        key_ = self.w_k(key)
        value_ = self.w_v(value)

        scores = torch.matmul(query_, key_.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))  # b*400*10

        # mask = mask.unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)    
        # mask = mask.float()
        # mask = (1.0 - mask) * -10000.0

        mask = mask.unsqueeze(2).repeat(1, 1, 10).float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask

        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)

        output = unshape(torch.matmul(probs, value_))
        output = self.final_linear(output)  # nn.Linear(768.768)

        return output
