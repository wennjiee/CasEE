
from func_load_bigram_data import *
from pathlib import Path

# newfile = open('C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/datasets/FewFC/data/new_train.json','a+',encoding='utf-8')
# with open('C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/datasets/FewFC/data/train.json','r',encoding='utf-8') as f:
#     lines = f.readlines()
#     for line in lines:
#         line  = line + '\n'
#         newfile.write(line)
# newfile.close()


def load_bigram_():

    print('begin')

    raw_dataset_cache_name = os.path.join('cache',  'fewfc_raw_cache')
    raw_dataset_cache_name = Path(raw_dataset_cache_name).as_posix()
    print('raw_cache_name路径:{}' ,format(raw_dataset_cache_name))
    refresh_data = False

    datasets, vocabs, embeddings = load_fewfc(
        _cache_fp = raw_dataset_cache_name,
        _refresh = refresh_data,
        path=fewfc_path,
        unigram_embedding_path=yangjie_rich_pretrain_unigram_path,
        bigram_embedding_path=yangjie_rich_pretrain_bigram_path,  # 一元、多元数据集
        index_token=False,
        char_min_freq=1,  # char_min_freq=1
        bigram_min_freq=1,  # bigram_min_freq=1
        only_train_min_freq=True  # only_train_min_freq=true
    )

    print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))
    # w_list 为文件ctb.50d.vec中所有的词 即每一行第一个元素
    refresh_data = False
    w_list = load_yangjie_rich_pretrain_word_list(embedding_path=yangjie_rich_pretrain_word_path,
                                                  drop_characters=True,
                                                  _refresh=refresh_data,
                                                  _cache_fp='cache/yj'
                                                  )
    print('词表loaded')

    cache_name = os.path.join('cache', '_lattice')
    cache_name = Path(cache_name).as_posix()
    print('cache_name路径:{}' ,format(cache_name))
    datasets, vocabs, embeddings = equip_with_lexicon(
                                                      datasets, vocabs, embeddings,
                                                      w_list, yangjie_rich_pretrain_word_path, # ctb.50d.vec
                                                      only_lexicon_in_train=False, # false
                                                      word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                      number_normalized=0, # number_normalized=0
                                                      lattice_min_freq=1, # lattice_min_freq=1
                                                      only_train_min_freq=True) # only_train_min_freq=true
    # print(embeddings['lattice'](torch.LongTensor([vocabs['lattice'].to_index('我们')])))

    print('loaded')
    sets = {}
    for k ,v in datasets.items():
        tmp = {}
        L = len(datasets[k].field_arrays['code'])
        for i in range(L):
            codeLength =  len(datasets[k].field_arrays['code'][i][0])
            code = datasets[k].field_arrays['code'][i][0][1:codeLength -2]
            tmp[code] = i
        sets[k] = tmp
    print('over')

    return sets, datasets, vocabs, embeddings