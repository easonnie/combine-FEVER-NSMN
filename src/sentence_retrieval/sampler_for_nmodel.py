"""
This file contains scripts to build or sample data for neural sentence selector.

Neural sentence selector aimed to fine-select sentence for NLI models since NLI models are sensitive to data.
"""

import json

from nn_doc_retrieval.disabuigation_training import trucate_item
from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format
from utils import fever_db, common, c_scorer
from utils.common import load_jsonl
from tqdm import tqdm
import config
from utils.tokenize_fever import easy_tokenize
import utils.check_sentences
import itertools
import numpy as np


def get_full_list(tokenized_data_file, additional_data_file, pred=False, top_k=None):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    cursor = fever_db.get_cursor()
    d_list = load_jsonl(tokenized_data_file)

    if not isinstance(additional_data_file, list):
        additional_d_list = load_jsonl(additional_data_file)
    else:
        additional_d_list = additional_data_file

    if top_k is not None:
        print("Upstream document number truncate to:", top_k)
        trucate_item(additional_d_list, top_k=top_k)

    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    full_data_list = []

    for item in tqdm(d_list):
        doc_ids = additional_data_dict[item['id']]["predicted_docids"]

        if not pred:
            if item['evidence'] is not None:
                e_list = utils.check_sentences.check_and_clean_evidence(item)
                all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
            else:
                all_evidence_set = None
            # print(all_evidence_set)
            r_list = []
            id_list = []

            if all_evidence_set is not None:
                for doc_id, ln in all_evidence_set:
                    _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                    r_list.append(text)
                    id_list.append(doc_id + '(-.-)' + str(ln))

        else:            # If pred, then reset to not containing ground truth evidence.
            all_evidence_set = None
            r_list = []
            id_list = []

        for doc_id in doc_ids:
            cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_r_list)):
                if cur_id_list[i] in id_list:
                    continue
                else:
                    r_list.append(cur_r_list[i])
                    id_list.append(cur_id_list[i])

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            sent_item['query'] = item['claim']

            if 'label' in item.keys():
                sent_item['claim_label'] = item['label']

            full_data_list.append(sent_item)

    return full_data_list


def get_full_list_from_list_d(tokenized_data_file, additional_data_file, pred=False, top_k=None):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    cursor = fever_db.get_cursor()
    d_list = tokenized_data_file

    additional_d_list = additional_data_file

    if top_k is not None:
        print("Upstream document number truncate to:", top_k)
        trucate_item(additional_d_list, top_k=top_k)

    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    full_data_list = []

    for item in tqdm(d_list):
        doc_ids = additional_data_dict[item['id']]["predicted_docids"]

        if not pred:
            if item['evidence'] is not None:
                e_list = utils.check_sentences.check_and_clean_evidence(item)
                all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
            else:
                all_evidence_set = None
            # print(all_evidence_set)
            r_list = []
            id_list = []

            if all_evidence_set is not None:
                for doc_id, ln in all_evidence_set:
                    _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
                    r_list.append(text)
                    id_list.append(doc_id + '(-.-)' + str(ln))

        else:            # If pred, then reset to not containing ground truth evidence.
            all_evidence_set = None
            r_list = []
            id_list = []

        for doc_id in doc_ids:
            cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_r_list)):
                if cur_id_list[i] in id_list:
                    continue
                else:
                    r_list.append(cur_r_list[i])
                    id_list.append(cur_id_list[i])

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            sent_item['query'] = item['claim']
            full_data_list.append(sent_item)

    return full_data_list


def get_additional_list(tokenized_data_file, additional_data_file,
                        item_key='prioritized_docids_aside', top_k=6):
    """
    This method will select all the sentence from upstream doc retrieval and label the correct evident as true
    :param item_key: The item that specify the additional prioritized document ids.
    :param tokenized_data_file: Remember this is tokenized data with original format containing 'evidence'
    :param additional_data_file:    This is the data after document retrieval.
                                    This file need to contain *"predicted_docids"* field.
    :return:
    """
    cursor = fever_db.get_cursor()
    d_list = load_jsonl(tokenized_data_file)

    additional_d_list = load_jsonl(additional_data_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[int(add_item['id'])] = add_item

    full_data_list = []

    for item in tqdm(d_list):
        doc_ids_p_list = additional_data_dict[int(item['id'])][item_key]
        doc_ids = list(set([k for k, v in sorted(doc_ids_p_list, key=lambda x: (-x[1], x[0]))][:top_k]))

        # if not pred:
        #     if item['evidence'] is not None:
        #         e_list = utils.check_sentences.check_and_clean_evidence(item)
        #         all_evidence_set = set(itertools.chain.from_iterable([evids.evidences_list for evids in e_list]))
        #     else:
        #         all_evidence_set = None
        #     # print(all_evidence_set)
        #     r_list = []
        #     id_list = []
        #
        #     if all_evidence_set is not None:
        #         for doc_id, ln in all_evidence_set:
        #             _, text, _ = fever_db.get_evidence(cursor, doc_id, ln)
        #             r_list.append(text)
        #             id_list.append(doc_id + '(-.-)' + str(ln))
        #
        # else:            # If pred, then reset to not containing ground truth evidence.

        all_evidence_set = None
        r_list = []
        id_list = []

        for doc_id in doc_ids:
            cur_r_list, cur_id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
            # Merging to data list and removing duplicate
            for i in range(len(cur_r_list)):
                if cur_id_list[i] in id_list:
                    continue
                else:
                    r_list.append(cur_r_list[i])
                    id_list.append(cur_id_list[i])

        assert len(id_list) == len(set(id_list))  # check duplicate
        assert len(r_list) == len(id_list)

        zipped_s_id_list = list(zip(r_list, id_list))
        # Sort using id
        # sorted(evidences_set, key=lambda x: (x[0], x[1]))
        zipped_s_id_list = sorted(zipped_s_id_list, key=lambda x: (x[1][0], x[1][1]))

        all_sent_list = convert_to_formatted_sent(zipped_s_id_list, all_evidence_set, contain_head=True,
                                                  id_tokenized=True)
        cur_id = item['id']
        for i, sent_item in enumerate(all_sent_list):
            sent_item['selection_id'] = str(cur_id) + "<##>" + str(sent_item['sid'])
            # selection_id is '[item_id<##>[doc_id]<SENT_LINE>[line_number]'
            sent_item['query'] = item['claim']
            full_data_list.append(sent_item)

    return full_data_list


def convert_to_formatted_sent(zipped_s_id_list, evidence_set, contain_head=True, id_tokenized=True):
    sent_list = []
    for sent, sid in zipped_s_id_list:
        sent_item = dict()

        cur_sent = sent
        doc_id, ln = sid.split('(-.-)')[0], int(sid.split('(-.-)')[1])
        # print(sent, doc_id, ln)
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
            # sid is '[doc_id]<SENT_LINE>[line_number]'
            if evidence_set is not None:
                if (doc_id, ln) in evidence_set:
                    sent_item['selection_label'] = "true"
                else:
                    sent_item['selection_label'] = "false"
            else:
                sent_item['selection_label'] = "hidden"

            sent_list.append(sent_item)
        else:
            sent_list.append(sent_item)

    # for s in sent_list:
    # print(s['text'][:20], s['selection_label'])

    return sent_list


def navie_results_builder_for_sanity_check(org_data_file, full_sent_list):
    """
    :param org_data_file:
    :param full_sent_list: append full_sent_score list to evidence of original data file
    :return:
    """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict = dict()
    print("Build selected sentences file")
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']    # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        if org_id in augmented_dict:
            augmented_dict[org_id].append(sent_item)
        else:
            augmented_dict[org_id] = [sent_item]

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = [] # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])]
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['selection_label'] == "true":
                    cur_predicted_sentids.append(sent_i['sid'])

        item['predicted_sentids'] = cur_predicted_sentids
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        item['predicted_label'] = item['label']

    return d_list


def post_filter(d_list, keep_prob=0.75, seed=12):
    np.random.seed(seed)
    r_list = []
    for item in d_list:
        if item['selection_label'] == 'false':
            if np.random.random(1) >= keep_prob:
                continue
        r_list.append(item)
    return r_list

# def get_formatted_sent(cursor, doc_id):
#     fever_db.get_all_sent_by_doc_id(cursor, doc_id)


if __name__ == '__main__':
    # full_list = get_full_list(config.T_FEVER_DEV_JSONL,
    #                           config.RESULT_PATH / "doc_retri/2018_07_04_21:56:49_r/dev.jsonl",
    #                           pred=True)
    # full_list = get_full_list(config.T_FEVER_TRAIN_JSONL,
    # config.RESULT_PATH / "doc_retri/2018_07_04_21:56:49_r/train.jsonl")
    train_upstream_file = config.RESULT_PATH / "doc_retri/2018_07_04_21:56:49_r/train.jsonl"
    complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False)
    filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=0.5, seed=12)

    full_list = complete_upstream_train_data
    # full_list = post_filter(full_list, keep_prob=0.5)
    print(len(full_list))
    print(len(filtered_train_data))
    count_hit = 0
    for item in full_list:
        # print(item)
        if item['selection_label'] == 'true':
            count_hit += 1

    print(count_hit, len(full_list), count_hit / len(full_list))

    # d_list = navie_results_builder_for_sanity_check(config.T_FEVER_DEV_JSONL, full_list)
    # # d_list = navie_results_builder_for_sanity_check(config.T_FEVER_TRAIN_JSONL, full_list)
    # eval_mode = {'check_sent_id_correct': True, 'standard': True}
    # print(c_scorer.fever_score(d_list, config.T_FEVER_DEV_JSONL, mode=eval_mode, verbose=False))
    #
    # total = len(d_list)
    # hit = eval_mode['check_sent_id_correct_hits']
    # tracking_score = hit / total
    # print("Tracking:", tracking_score)

    # for item in full_list:
    #     print(item)