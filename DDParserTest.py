from ddparser import DDParser
import json
from tqdm import tqdm
ddp = DDParser(tree=True)

dic = {'SBV':'主','VOB':'动','POB':'介','ADV':'状','CMP':'补',
       'ATT':'定','F':'位','COO':'并','DBL':'兼','DOB':'双',
       'VV':'连','IC':'子','MT':'虚','HED':'核'}

def ddparser_str(sentence):
    result = ddp.parse(sentence)
    word = result[0]['word']
    head = result[0]['head']
    deprel = result[0]['deprel']
    sentence_label = []
    head_label = []
    for i in range(len(word)):
        w = word[i]
        for j in range(len(w)):
            sentence_label.append(dic[deprel[i]])
            head_label.append(head[i])
    return sentence_label, head_label


input_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/datasets/FewFC/data/test.json'
output_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/datasets/FewFC/data/test_syn.json'
outfile = open(output_path, 'w', encoding='utf-8')
cnt = 0
for line in tqdm(open(input_path, 'r', encoding='utf-8')):
    cnt = cnt + 1
    raw_data = json.loads(line)
    sentence = raw_data['content']
    sentence_label, head_label = ddparser_str(sentence)
    raw_data['sentence_label'] = sentence_label
    raw_data['head_label'] = head_label
    json_str = json.dumps(raw_data, ensure_ascii=False)
    outfile.write(json_str)
    outfile.write('\n')
    #print(cnt)
outfile.close()
print('convert over')

# sentence = "嘉麟杰(行情002486,诊股)日前正在筹划作价2.6亿元收购北极光电,切入光通信领域。"
# result = ddp.parse(sentence)
# print(result)
# word = result[0]['word']
# head = result[0]['head']
# deprel = result[0]['deprel']
# sentence_label = []
# head_label = []
# for i in range(len(word)):
#     w = word[i]
#     for j in range(len(w)):
#         sentence_label.append(deprel[i])
#         head_label.append(head[i])
# print("ddparser end")


# 由此,二审判决撤销了一审中的相关刑事判决,
# 改判区志航有期徒刑三年六个月,
# 并处罚金人民币320万元;
# 改判袁园有期徒刑二年,并处罚金人民币15万元。"
# [{'word': [
# '由此', ',', '二审', '判决', '撤销', '了', '一审', '中', '的', '相关', '刑事', '判决', ',',
# '改判', '区', '志航', '有期徒刑', '三年', '六个', '月', ',',
# '并处', '罚金', '人民币', '320万元', ';',
# '改判', '袁园', '有期徒刑', '二年', ',', '并处', '罚金', '人民币', '15万元', '。'],
# 'head': [5, 1, 4, 5, 0, 5, 8, 12, 8, 12, 12, 5, 5, 撤销
# 5, 16, 17, 14, 17, 20, 18, 14, 撤销
# 14, 24, 25, 22, 22, 改判
# 22, 29, 27, 29, 27, 22, 32, 32, 34, 5], 并处
# 'deprel': [
# 'ADV由此<-撤销', 'MT', 'ATT', 'SBV', 'HED', 'MT', 'ATT', 'ATT', 'MT', 'ATT', 'ATT', 'VOB', 'MT',
# 'COO改判<-撤销', 'ATT', 'ATT', 'VOB', 'ATT', 'ATT', 'ATT', 'MT',
# 'COO并处<-改判', 'ATT', 'ATT', 'VOB', 'MT',
# 'COO改判<-并处', 'ATT', 'VOB', 'ATT', 'MT', 'COO', 'VOB', 'VOB', 'ATT', 'MT']}]
# result = ddp.parse("百度是一家高科技公司")

# [{'word': ['百度', '是', '一家', '高科技', '公司'],
# 'head': [2, 0, 5, 5, 2],
# 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]
