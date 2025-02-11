import copy
from fastNLP import Vocabulary, DataSet
import os
from fastNLP import cache_results
from fastNLP.io.loader import ConllLoader
from functools import partial
import numpy
from fastNLP.embeddings import BertEmbedding
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding
import collections

yangjie_rich_pretrain_unigram_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/NeuralSegmentation/gigaword_chn.all.a2b.uni.ite50.vec'
yangjie_rich_pretrain_bigram_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/NeuralSegmentation/gigaword_chn.all.a2b.bi.ite50.vec'
yangjie_rich_pretrain_word_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/NeuralSegmentation/ctb.50d.vec'

# this path is for the output of preprocessing
yangjie_rich_pretrain_char_and_word_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/NeuralSegmentation/yangjie_word_char_mix.txt'

fewfc_path = 'C:/Users/qc/Desktop/AllGitHubCode/CasEE-main/datasets/FewFC/data'


# load ctb.50d.vec
@cache_results(_cache_fp='cache/pretrain_word_list', _refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r', encoding='utf-8')
    lines = f.readlines()
    w_list = []
    # strip()去除字符串左右两边的空格
    # 当不给split函数传递任何参数时，分隔符sep会采用任意形式的空白字符：空格、tab、换行、回车以及formfeed
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)
    if drop_characters:  # 只使用词
        w_list = list(filter(lambda x: len(x) != 1, w_list))  # 只需要长度不为1的元素
    return w_list


@cache_results(_cache_fp='cache/fewfc_uni_bi', _refresh=False)
def load_fewfc(path, unigram_embedding_path=None, bigram_embedding_path=None, index_token=True,
               char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0, char_word_dropout=0.01, label='all'):
    loader = ConllLoader(['id', 'code', 'content_', 'chars'])
    train_path = os.path.join(path, 'new_train.json')
    dev_path = os.path.join(path, 'new_dev.json')
    test_path = os.path.join(path, 'new_test.json')

    paths = {'train': train_path, 'dev': dev_path, 'test': test_path}

    datasets = {}
    for k, v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    for k, v in paths.items():
        for i in range(len(datasets[k].field_arrays['chars'])):
            str = datasets[k].field_arrays['chars'][i][0]
            datasets[k].field_arrays['chars'][i] = list(str)

    for k, v in datasets.items():
        v.apply_field(get_bigrams, 'chars', 'bigrams')  # 两两生成词语

    for k, v in datasets.items():
        v.add_seq_len('chars', new_field_name='seq_len')
    # 创建词表 运行后输出0代表pad，1代表unk，2代表，，3代表的...   char_vocab.to_index('的')=3
    vocabs = {}
    bigram_vocab = Vocabulary()
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',no_create_entry_dataset=[datasets['dev'], datasets['test']])

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    # print(unigram_embedding(torch.LongTensor([char_vocab.to_index('我')])))

    return datasets, vocabs, embeddings


def get_bigrams(words):
    result = []
    for i, w in enumerate(words):
        if i != len(words) - 1:
            result.append(words[i] + words[i + 1])
        else:
            result.append(words[i] + '<end>')
    return result

@cache_results(_cache_fp='cache/lexicons', _refresh=False)
def equip_with_lexicon(datasets, vocabs, embeddings, w_list, word_embedding_path=None,
                       only_lexicon_in_train=False, word_char_mix_embedding_path=None,
                       number_normalized=False,
                       lattice_min_freq=1, only_train_min_freq=0):
    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)
        return result

    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x: x[2], lexicons))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e
        return pos_e

    a = DataSet()
    w_trie = Trie()  # 字典查找树
    for w in w_list:  # w_list是所有词的集合
        w_trie.insert(w)


    for k, v in datasets.items():
        # lexicons为[0, 1, '科技'], [2, 3, '全方'], [2, 4, '全方位'], [3, 4, '方位']...
        v.apply_field(partial(get_skip_path, w_trie=w_trie), 'chars', 'lexicons') # partial接收get_skip函数 w_trie作为默认值
        v.apply_field(copy.copy, 'chars', 'raw_chars') # raw_chars = copy.copy(chars)

        # lex_num代表数据集每句话的词语个数 第一句话10个词 第二句话5个词...
        v.add_seq_len('lexicons', 'lex_num') # 将使用len()直接对field_name中每个元素作用，将其结果作为seqence length, 并放入seq_len这个field。
        # lex_start词的开始位置0 2 2 3 5 7，lex_end词的结束位置1 3 4 4 6 8
        v.apply_field(lambda x: list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')

    print('loaded one')
    for k, v in datasets.items():
        v.apply(concat, new_field_name='lattice')
        v.set_input('lattice')
        v.apply(get_pos_s, new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s', 'pos_e')

    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)  # 依次增加序列中词在词典中的出现频率
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'], field_name='lattice',
                               no_create_entry_dataset=[v for k, v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab

    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab, word_embedding_path, word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path, word_dropout=0.01,
                                            min_freq=lattice_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    # vocabs['char'].index_dataset(*(datasets.values()), field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(*(datasets.values()), field_name='bigrams', new_field_name='bigrams')
    vocabs['lattice'].index_dataset(*(datasets.values()), field_name='lattice', new_field_name='lattice')


    return datasets, vocabs, embeddings


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_w = False


# Trie 单词查找树
class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):
        current = self.root
        for c in w:
            current = current.children[c]
        current.is_w = True

    def search(self, w):
        '''
        :param w:
        :return:
        -1:not w route
        0:subroute but not word
        1:subroute and word
        '''
        current = self.root
        for c in w:
            current = current.children.get(c)
            if current is None:
                return -1
        if current.is_w:
            return 1
        else:
            return 0

    def get_lexicon(self, sentence):
        result = []
        for i in range(len(sentence)):
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.append([i, j, sentence[i:j + 1]])
        return result
