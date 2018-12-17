import json

import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from utils import fever_db, text_clean
from tqdm import tqdm


class DocIdDict(object):
    def __init__(self):
        self.tokenized_doc_id_dict = None

    def load_dict(self):
        if self.tokenized_doc_id_dict is None:
            self.tokenized_doc_id_dict = json.load(open(config.TOKENIZED_DOC_ID, encoding='utf-8', mode='r'))

    def clean(self):
        self.tokenized_doc_id_dict = None


# global tokenized_doc_id_dict
# tokenized_doc_id_dict = None
global_doc_id_object = DocIdDict()


def e_tokenize(text, tok):
    return tok.tokenize(text_clean.normalize(text))


def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')


def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def tokenize_doc_id(doc_id, tokenizer):
    # path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    # print(path_stanford_corenlp_full_2017_06_09)
    #
    # drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    # tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
    tokenized_doc_id = e_tokenize(doc_id_natural_format, tokenizer)
    t_doc_id_natural_format = tokenized_doc_id.words()
    lemmas = tokenized_doc_id.lemmas()
    return t_doc_id_natural_format, lemmas


def doc_id_to_tokenized_text(doc_id, including_lemmas=False):
    # global tokenized_doc_id_dict
    global_doc_id_object.load_dict()
    tokenized_doc_id_dict = global_doc_id_object.tokenized_doc_id_dict

    if tokenized_doc_id_dict is None:
        tokenized_doc_id_dict = json.load(open(config.TOKENIZED_DOC_ID, encoding='utf-8', mode='r'))

    if including_lemmas:
        return tokenized_doc_id_dict[doc_id]['words'], tokenized_doc_id_dict[doc_id]['lemmas']

    return ' '.join(tokenized_doc_id_dict[doc_id]['words'])


if __name__ == '__main__':
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    print(path_stanford_corenlp_full_2017_06_09)
    # #
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])
    #
    id_to_natural_id_dict = dict()
    d_list = load_jsonl("/Users/Eason/RA/FunEver/data/id_dict.jsonl")
    for item in tqdm(d_list):
        doc_id = item['docid']
        id_to_natural_id_dict[doc_id] = dict()
        words, lemmas = tokenize_doc_id(doc_id, tok)
        id_to_natural_id_dict[doc_id]['words'] = words
        id_to_natural_id_dict[doc_id]['lemmas'] = lemmas

    with open(config.DATA_ROOT / "tokenized_doc_id.json", encoding='utf-8', mode='w') as out_f:
        json.dump(id_to_natural_id_dict, out_f)
    # print("Yes")
    # print(doc_id_to_tokenized_text('ABC'))
    # print(doc_id_to_tokenized_text('ABC'))



