# This is for utest for whether training sentences contain all the sentences.
from typing import Tuple, List, Dict

import utils.common
from tqdm import tqdm
from utils import c_scorer, check_sentences, fever_db, common
import config
from simi_sampler_nli_v0 import simi_sampler
from utils.tokenize_fever import easy_tokenize

# utils.common.load_jsonl(
#         "/Users/Eason/RA/FunEver/results/sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")


def complete_sentence_for_training_set(
        train_sent_list=None,
        training_file=config.T_FEVER_TRAIN_JSONL, cursor=fever_db.get_cursor()):

    training_list = utils.common.load_jsonl(training_file)

    selection_dict = simi_sampler.paired_selection_score_dict(train_sent_list)

    total = 0
    hit = 0

    remain_sent_list: List[Dict] = []

    for item in tqdm(training_list):
        item_id: int = int(item['id'])
        e_list = check_sentences.check_and_clean_evidence(item)
        for evidences in e_list:
            for doc_id, ln in evidences:
                ssid: Tuple[int, str, int] = (item_id, doc_id, ln)
                if ssid in selection_dict:
                    hit += 1
                else:
                    data_point = create_one_item(item, item_id, doc_id, ln, cursor, True, True)
                    remain_sent_list.append(data_point)

                total += 1

    print(hit, total, hit / total)

    cursor.close()

    return remain_sent_list


def create_one_item(item: Dict, item_id: int, doc_id: str, ln: int, cursor, contain_head=True, id_tokenized=False):
    sent_item = dict()

    _, cur_sent, _ = fever_db.get_evidence(cursor, doc_id, ln)
    assert cur_sent is not None and cur_sent != ''

    sent = cur_sent
    if contain_head:
        if not id_tokenized:
            doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
            t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
        else:
            t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

        if ln != 0 and t_doc_id_natural_format.lower() not in sent.lower():
            cur_sent = f"{t_doc_id_natural_format} <t> " + sent

        sent_item['text'] = cur_sent
        sent_item['sid'] = doc_id + c_scorer.SENT_LINE + str(ln)

        sent_item['selection_label'] = "true"

        cur_id = item['id']
        assert cur_id == item_id
        sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
        sent_item['query'] = item['claim']

    return sent_item


if __name__ == '__main__':
    # remaining_training_list = complete_sentence_for_training_set()
    # print(len(remaining_training_list))
    # common.save_jsonl(remaining_training_list, "/Users/Eason/RA/FunEver/results/sent_retri_nn/remaining_training_cache/train.jsonl")

    remaining_training_list = complete_sentence_for_training_set(
        train_sent_list=utils.common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl"),
        training_file=config.T_FEVER_DEV_JSONL
    )
    print(len(remaining_training_list))
    common.save_jsonl(remaining_training_list,
                      "/Users/Eason/RA/FunEver/results/sent_retri_nn/remaining_training_cache/dev_s.jsonl")

    # train_sent_list = utils.common.load_jsonl("/Users/Eason/RA/FunEver/results/sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")
    # train_sent_id_dict = set()
    #
    # training_list = utils.common.load_jsonl(config.T_FEVER_DEV_JSONL)
    #
    # selection_dict = simi_sampler.paired_selection_score_dict(train_sent_list)
    #
    # total = 0
    # hit = 0
    #
    # for item in tqdm(training_list):
    #     new_item: List[Dict] = dict()
    #
    #     item_id: int = int(item['id'])
    #     e_list = check_sentences.check_and_clean_evidence(item)
    #     for evidences in e_list:
    #         for doc_id, ln in evidences:
    #             ssid: Tuple[int, str, int] = (item_id, doc_id, ln)
    #             if ssid in selection_dict:
    #                 hit += 1
    #             total += 1
    #
    #     new_id: str = f"{item_id}<##>{doc_id}{c_scorer.SENT_LINE}{ln}"
    #
    # print(hit, total, hit / total)