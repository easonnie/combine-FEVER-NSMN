# This is for utest for whether training sentences contain all the sentences.
from typing import Tuple

import utils.common
from tqdm import tqdm
from utils import c_scorer, check_sentences
import config
from simi_sampler_nli_v0 import simi_sampler


if __name__ == '__main__':
    train_sent_list = utils.common.load_jsonl("/Users/Eason/RA/FunEver/results/sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")
    # train_sent_list = utils.common.load_jsonl("/Users/Eason/RA/FunEver/results/sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")

    remaining_sent_list = utils.common.load_jsonl("/Users/Eason/RA/FunEver/results/sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")

    train_sent_id_dict = set()

    training_list = utils.common.load_jsonl(config.T_FEVER_TRAIN_JSONL)
    # training_list = utils.common.load_jsonl(config.T_FEVER_DEV_JSONL)

    # for item in tqdm(train_sent_list):
    #     selection_id = item['selection_id']
    #     item_id = int(selection_id.split('<##>')[0])
    #     sentid = selection_id.split('<##>')[1]
    #     doc_id = sentid.split(c_scorer.SENT_LINE)[0]
    #     ln = int(sentid.split(c_scorer.SENT_LINE)[1])
    #
    #     ssid = (item_id, doc_id, ln)
    #     print(ssid)
    #     train_sent_id_dict.add(ssid)
    selection_dict = simi_sampler.paired_selection_score_dict(train_sent_list)
    selection_dict = simi_sampler.paired_selection_score_dict(remaining_sent_list, selection_dict)

    # for k, v in selection_dict.items():
    #     print(k, v)

    total = 0
    hit = 0

    for item in tqdm(training_list):
        item_id = int(item['id'])
        e_list = check_sentences.check_and_clean_evidence(item)
        for evidences in e_list:
            for doc_id, ln in evidences:
                ssid: Tuple[int, str, int] = (item_id, doc_id, ln)
                if ssid in selection_dict:
                    assert item['claim'] == selection_dict[ssid]['claim']
                    hit += 1
                total += 1

    print(hit, total, hit / total, total - hit)

            # for doc_id, ln in evidences:
            #     if (item_id, doc_id, ln) not in selection_dict:
            #         print(item)

