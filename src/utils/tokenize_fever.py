import json

import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from utils import fever_db, text_clean
from tqdm import tqdm


def easy_tokenize(text, tok):
    return tok.tokenize(text_clean.normalize(text)).words()


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


def tokenized_claim(in_file, out_file):
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    print(path_stanford_corenlp_full_2017_06_09)
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])

    d_list = load_jsonl(in_file)
    for item in tqdm(d_list):
        item['claim'] = ' '.join(easy_tokenize(item['claim'], tok))

    save_jsonl(d_list, out_file)


def tokenized_claim_list(in_list):
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])

    for item in tqdm(in_list):
        item['claim'] = ' '.join(easy_tokenize(item['claim'], tok))

    return in_list


if __name__ == '__main__':
    # tokenized_claim(config.FEVER_DEV_JSONL, config.DATA_ROOT / "tokenized_fever/dev.jsonl")
    tokenized_claim(config.FEVER_TRAIN_JSONL, config.DATA_ROOT / "tokenized_fever/train.jsonl")