import json
import random
import copy

from sample_for_nli.tf_idf_sample_v1_0 import convert_evidence2scoring_format
from sentence_retrieval.nn_postprocess_ablation import score_converter_scaled
from utils import fever_db, check_sentences
import config
from tqdm import tqdm
from utils import c_scorer, text_clean, common
from collections import Counter
import numpy as np

from typing import Dict, List, Tuple
from tqdm import tqdm

from utils.tokenize_fever import easy_tokenize


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def paired_selection_score_dict(sent_list: List[Dict],
                                selection_dict: Dict[Tuple[int, str, int], Dict] = None) -> Dict[
    Tuple[int, str, int], Dict]:
    if selection_dict is None:
        selection_dict: Dict[tuple, Dict] = dict()

    for item in tqdm(sent_list):
        selection_id: str = item['selection_id']
        item_id: int = int(selection_id.split('<##>')[0])
        sentid: str = selection_id.split('<##>')[1]
        doc_id: str = sentid.split(c_scorer.SENT_LINE)[0]
        ln: int = int(sentid.split(c_scorer.SENT_LINE)[1])

        score: float = float(item['score'])
        prob: float = float(item['prob'])
        claim: str = item['query']

        ssid: Tuple[int, str, int] = (item_id, doc_id, ln)
        if ssid in selection_dict:
            assert claim == selection_dict[ssid]['claim']
            error_rate_score = score - float(selection_dict[ssid]['score'])
            error_rate_prob = prob - float(selection_dict[ssid]['prob'])
            assert error_rate_prob < 0.01
        else:
            selection_dict[ssid] = dict()
            selection_dict[ssid]['score'] = score
            selection_dict[ssid]['prob'] = prob
            selection_dict[ssid]['claim'] = claim

    return selection_dict


# This function is build on top of the previous function that provide document relatedness score
# Created on Aug 30, 2018
def paired_selection_score_dict_for_doc(doc_list: List[Dict],
                                        selection_dict: Dict[Tuple[int, str], Dict] = None) -> Dict[
    Tuple[int, str], Dict]:
    if selection_dict is None:
        selection_dict: Dict[tuple, Dict] = dict()

    for item in tqdm(doc_list):
        selection_id: str = item['selection_id']
        item_id: int = int(selection_id.split('###')[0])
        doc_id: str = selection_id.split('###')[1]

        # doc_id: str = sentid.split(c_scorer.SENT_LINE)[0]
        # ln: int = int(sentid.split(c_scorer.SENT_LINE)[1])

        score: float = float(item['score'])
        prob: float = float(item['prob'])
        claim: str = item['text']

        dssid: Tuple[int, str] = (item_id, doc_id)
        if dssid in selection_dict:
            assert claim == selection_dict[dssid]['claim']
            error_rate_score = score - float(selection_dict[dssid]['score'])
            error_rate_prob = prob - float(selection_dict[dssid]['prob'])
            if error_rate_prob > 0.02:
                print(dssid)
                print(item)
                print(selection_dict[dssid])

            assert error_rate_prob < 0.05
        else:
            selection_dict[dssid] = dict()
            selection_dict[dssid]['score'] = score
            selection_dict[dssid]['prob'] = prob
            selection_dict[dssid]['claim'] = claim

    return selection_dict


def threshold_sampler(org_data_file, full_sent_list, prob_threshold=0.5, logist_threshold=None, top_n=5):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        if org_id in augmented_dict:
            # change some logic to remove duplicate.
            augmented_dict[org_id].append(sent_item)
        else:
            augmented_dict[org_id] = [sent_item]

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])]
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def threshold_sampler_insure_unique(org_data_file, full_sent_list, prob_threshold=0.5, logist_threshold=None, top_n=5):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict: Dict[int, Dict[str, Dict]] = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        # doc_id = remain_str.split(c_scorer.SENT_LINE)[0]
        # ln = int(remain_str.split(c_scorer.SENT_LINE)[1])
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
            else:
                print("Exist")
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def threshold_sampler_insure_unique_list(org_data_file, full_sent_list, prob_threshold=0.5, logist_threshold=None, top_n=5):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    d_list = org_data_file
    augmented_dict: Dict[int, Dict[str, Dict]] = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        # doc_id = remain_str.split(c_scorer.SENT_LINE)[0]
        # ln = int(remain_str.split(c_scorer.SENT_LINE)[1])
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
            else:
                print("Exist")
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
        item['predicted_sentids'] = [sid for sid, _, _ in item['scored_sentids']][:top_n]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def threshold_sampler_insure_unique_merge(org_data_file, full_sent_list, prob_threshold=0.5,
                                          logist_threshold=None, top_n=5, add_n=1):
    """
    Providing samples to the Training set by a probability threshold on the upstream selected sentences.
    """
    if not isinstance(org_data_file, list):
        d_list = common.load_jsonl(org_data_file)
    else:
        d_list = org_data_file
    augmented_dict: Dict[int, Dict[str, Dict]] = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        remain_str = selection_id.split('<##>')[1]
        # doc_id = remain_str.split(c_scorer.SENT_LINE)[0]
        # ln = int(remain_str.split(c_scorer.SENT_LINE)[1])
        if org_id in augmented_dict:
            if remain_str not in augmented_dict[org_id]:
                augmented_dict[org_id][remain_str] = sent_item
        else:
            augmented_dict[org_id] = {remain_str: sent_item}

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            # print("Potential error?")
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])].values()
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= prob_threshold:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score'],
                                                  sent_i['prob']))  # Important sentences for scaling training. Jul 21.
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        cur_predicted_sentids = cur_predicted_sentids[:add_n]

        # if item['scored_sentids']
        if len(item['predicted_sentids']) >= 5:
            continue
        else:
            # item['scored_sentids'] = cur_predicted_sentids[:top_n]  # Important sentences for scaling training. Jul 21.
            item['predicted_sentids'].extend(
                [sid for sid, _, _ in cur_predicted_sentids if sid not in item['predicted_sentids']])
            item['predicted_sentids'] = item['predicted_sentids'][:top_n]
            item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])

        # item['predicted_label'] = item['label']  # give ground truth label

    return d_list


def sample_additional_data_for_item_v1_0(item, additional_data_dictionary):
    res_sentids_list = []
    flags = []

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_list = check_sentences.check_and_clean_evidence(item)
        current_id = item['id']
        assert current_id in additional_data_dictionary
        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        # additional_data_with_score = additional_data_dictionary[current_id]['scored_sentids']

        # print(len(additional_data))

        for evidences in e_list:
            # print(evidences)
            new_evidences = copy.deepcopy(evidences)
            n_e = len(evidences)
            if n_e < 5:
                current_sample_num = random.randint(0, 5 - n_e)
                random.shuffle(additional_data)
                for sampled_e in additional_data[:current_sample_num]:
                    doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
                    ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
                    new_evidences.add_sent(doc_ids, ln)

            if new_evidences != evidences:
                flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
                flags.append(flag)
                pass
            else:
                flag = "verifiable.eq.0"
                flags.append(flag)
                pass
            res_sentids_list.append(new_evidences)

        assert len(res_sentids_list) == len(e_list)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'

        e_list = check_sentences.check_and_clean_evidence(item)
        current_id = item['id']
        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        # print(len(additional_data))
        random.shuffle(additional_data)
        current_sample_num = random.randint(2, 5)
        raw_evidences_list = []
        for sampled_e in additional_data[:current_sample_num]:
            doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
            ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
            raw_evidences_list.append((doc_ids, ln))
        new_evidences = check_sentences.Evidences(raw_evidences_list)

        if len(new_evidences) == 0:
            flag = f"verifiable.eq.0"
            flags.append(flag)
            pass
        else:
            flag = f"not_verifiable.non_eq.{len(new_evidences)}"
            flags.append(flag)

        assert all(len(e) == 0 for e in e_list)
        res_sentids_list.append(new_evidences)
        assert len(res_sentids_list) == 1

    assert len(res_sentids_list) == len(flags)

    return res_sentids_list, flags


def sample_additional_data_for_item_v1_1(item, additional_data_dictionary):
    res_sentids_list = []
    flags = []

    if item['verifiable'] == "VERIFIABLE":
        assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
        e_list = check_sentences.check_and_clean_evidence(item)
        current_id = item['id']
        assert current_id in additional_data_dictionary
        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        # additional_data_with_score = additional_data_dictionary[current_id]['scored_sentids']

        # print(len(additional_data))

        for evidences in e_list:
            # print(evidences)
            new_evidences = copy.deepcopy(evidences)
            n_e = len(evidences)
            if n_e < 5:
                current_sample_num = random.randint(0, 5 - n_e)
                random.shuffle(additional_data)
                for sampled_e in additional_data[:current_sample_num]:
                    doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
                    ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
                    new_evidences.add_sent(doc_ids, ln)

            if new_evidences != evidences:
                flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
                flags.append(flag)
                pass
            else:
                flag = "verifiable.eq.0"
                flags.append(flag)
                pass
            res_sentids_list.append(new_evidences)

        assert len(res_sentids_list) == len(e_list)

    elif item['verifiable'] == "NOT VERIFIABLE":
        assert item['label'] == 'NOT ENOUGH INFO'

        e_list = check_sentences.check_and_clean_evidence(item)
        current_id = item['id']

        additional_data = additional_data_dictionary[current_id]['predicted_sentids']
        prioritized_additional_evidence_list = additional_data_dictionary[current_id]['scored_sentids']

        #  cur_predicted_sentids.append((sent_i['sid'], sent_i['score'], sent_i['prob']))
        certain_k = 2
        prioritized_additional_evidence_list = sorted(prioritized_additional_evidence_list, key=lambda x: -x[1])
        top_two_sent = [sid for sid, _, _ in prioritized_additional_evidence_list[:certain_k]]

        random.shuffle(additional_data)
        current_sample_num = random.randint(0, 2)
        raw_evidences_list = []

        # Debug
        # print(prioritized_additional_evidence_list)
        # print(top_two_sent)

        for sampled_e in top_two_sent + additional_data[:current_sample_num]:
            doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
            ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
            raw_evidences_list.append((doc_ids, ln))
        new_evidences = check_sentences.Evidences(raw_evidences_list)

        if len(new_evidences) == 0:
            flag = f"verifiable.eq.0"
            flags.append(flag)
            pass
        else:
            flag = f"not_verifiable.non_eq.{len(new_evidences)}"
            flags.append(flag)

        assert all(len(e) == 0 for e in e_list)
        res_sentids_list.append(new_evidences)
        assert len(res_sentids_list) == 1

        # Debug
        # print(res_sentids_list)

    assert len(res_sentids_list) == len(flags)

    return res_sentids_list, flags


# def sample_additional_data_for_item_v1_2(item, additional_data_dictionary):
#     """
#     Created on 04 Oct 2018 10:03:27
#     This method sample the upstream sentence file according to probability weights
#     :param item:
#     :param additional_data_dictionary:
#     :return:
#     """
#     res_sentids_list = []
#     flags = []
#
#     if item['verifiable'] == "VERIFIABLE":
#         assert item['label'] == 'SUPPORTS' or item['label'] == 'REFUTES'
#         e_list = check_sentences.check_and_clean_evidence(item)
#         current_id = item['id']
#         assert current_id in additional_data_dictionary
#         additional_data = additional_data_dictionary[current_id]['predicted_sentids']
#         # additional_data_with_score = additional_data_dictionary[current_id]['scored_sentids']
#
#         # print(len(additional_data))
#
#         for evidences in e_list:
#             # print(evidences)
#             new_evidences = copy.deepcopy(evidences)
#             n_e = len(evidences)
#             if n_e < 5:
#                 current_sample_num = random.randint(0, 5 - n_e)
#                 random.shuffle(additional_data)
#                 for sampled_e in additional_data[:current_sample_num]:
#                     doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
#                     ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
#                     new_evidences.add_sent(doc_ids, ln)
#
#             if new_evidences != evidences:
#                 flag = f"verifiable.non_eq.{len(new_evidences) - len(evidences)}"
#                 flags.append(flag)
#                 pass
#             else:
#                 flag = "verifiable.eq.0"
#                 flags.append(flag)
#                 pass
#             res_sentids_list.append(new_evidences)
#
#         assert len(res_sentids_list) == len(e_list)
#
#     elif item['verifiable'] == "NOT VERIFIABLE":
#         assert item['label'] == 'NOT ENOUGH INFO'
#
#         e_list = check_sentences.check_and_clean_evidence(item)
#         current_id = item['id']
#
#         additional_data = additional_data_dictionary[current_id]['predicted_sentids']
#         prioritized_additional_evidence_list = additional_data_dictionary[current_id]['scored_sentids']
#
#         #  cur_predicted_sentids.append((sent_i['sid'], sent_i['score'], sent_i['prob']))
#         certain_k = 2
#         prioritized_additional_evidence_list = sorted(prioritized_additional_evidence_list, key=lambda x: -x[1])
#         top_two_sent = [sid for sid, _, _ in prioritized_additional_evidence_list[:certain_k]]
#
#         random.shuffle(additional_data)
#         current_sample_num = random.randint(0, 2)
#         raw_evidences_list = []
#
#         # Debug
#         # print(prioritized_additional_evidence_list)
#         # print(top_two_sent)
#
#         for sampled_e in top_two_sent + additional_data[:current_sample_num]:
#             doc_ids = sampled_e.split(c_scorer.SENT_LINE)[0]
#             ln = int(sampled_e.split(c_scorer.SENT_LINE)[1])
#             raw_evidences_list.append((doc_ids, ln))
#         new_evidences = check_sentences.Evidences(raw_evidences_list)
#
#         if len(new_evidences) == 0:
#             flag = f"verifiable.eq.0"
#             flags.append(flag)
#             pass
#         else:
#             flag = f"not_verifiable.non_eq.{len(new_evidences)}"
#             flags.append(flag)
#
#         assert all(len(e) == 0 for e in e_list)
#         res_sentids_list.append(new_evidences)
#         assert len(res_sentids_list) == 1
#
#         # Debug
#         # print(res_sentids_list)
#
#     assert len(res_sentids_list) == len(flags)
#
#     return res_sentids_list, flags


def adv_simi_sample_with_prob_v1_0(input_file, additional_file, prob_dict_file, tokenized=False):
    cursor = fever_db.get_cursor()
    d_list = load_data(input_file)

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    sampled_data_list = []
    count = 0

    for item in tqdm(d_list):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, flags = sample_additional_data_for_item_v1_0(item, additional_data_dict)
        # print(flags)
        for i, (sampled_evidence, flag) in enumerate(zip(sampled_e_list, flags)):
            # Do not copy, might change in the future for error analysis
            # new_item = copy.deepcopy(item)
            new_item = dict()
            # print(new_item['claim'])
            # print(e_list)
            # print(sampled_evidence)
            # print(flag)
            evidence_text_list = evidence_list_to_text_list(
                cursor, sampled_evidence,
                contain_head=True, id_tokenized=tokenized)

            evidences = sorted(sampled_evidence, key=lambda x: (x[0], x[1]))
            item_id = int(item['id'])

            evidence_text_list_with_prob = []
            for text, (doc_id, ln) in zip(evidence_text_list, evidences):
                ssid = (int(item_id), doc_id, int(ln))
                if ssid not in prob_dict_file:
                    count += 1
                    print("Some sentence pair don't have 'prob'.")
                    prob = 0.5
                else:
                    prob = prob_dict_file[ssid]['prob']
                    assert item['claim'] == prob_dict_file[ssid]['claim']

                evidence_text_list_with_prob.append((text, prob))

            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                new_item['claim'] = ' '.join(easy_tokenize(item['claim']))

            new_item['evid'] = evidence_text_list_with_prob

            new_item['verifiable'] = item['verifiable']
            new_item['label'] = item['label']

            # print("C:", new_item['claim'])
            # print("E:", new_item['evid'])
            # print("L:", new_item['label'])
            # print()
            sampled_data_list.append(new_item)

    cursor.close()

    print(count)
    return sampled_data_list


# This function is build on top of the previous function that provide document relatedness score
# Created on Aug 30, 2018
def adv_simi_sample_with_prob_v1_0_with_doc(input_file, additional_file,
                                            prob_dict_file,
                                            prob_doc_dict_file,
                                            tokenized=False):
    cursor = fever_db.get_cursor()
    d_list = load_data(input_file)

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    sampled_data_list = []
    count = 0

    for item in tqdm(d_list):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, flags = sample_additional_data_for_item_v1_0(item, additional_data_dict)
        # print(flags)
        for i, (sampled_evidence, flag) in enumerate(zip(sampled_e_list, flags)):
            # Do not copy, might change in the future for error analysis
            # new_item = copy.deepcopy(item)
            new_item = dict()
            # print(new_item['claim'])
            # print(e_list)
            # print(sampled_evidence)
            # print(flag)
            evidence_text_list = evidence_list_to_text_list(
                cursor, sampled_evidence,
                contain_head=True, id_tokenized=tokenized)

            evidences = sorted(sampled_evidence, key=lambda x: (x[0], x[1]))
            item_id = int(item['id'])

            evidence_text_list_with_prob = []
            for text, (doc_id, ln) in zip(evidence_text_list, evidences):
                ssid = (int(item_id), doc_id, int(ln))
                if ssid not in prob_dict_file:
                    count += 1
                    print("Some sentence pair don't have 'prob'.")
                    prob = 0.5
                else:
                    prob = prob_dict_file[ssid]['prob']
                    assert item['claim'] == prob_dict_file[ssid]['claim']

                dssid = (item_id, doc_id)
                if dssid not in prob_doc_dict_file:
                    # print("Some sentence pair don't have 'prob'.")
                    doc_prob = 0.99
                else:
                    doc_prob = prob_doc_dict_file[dssid]['prob']
                    assert item['claim'] == prob_doc_dict_file[dssid]['claim']

                evidence_text_list_with_prob.append((text, prob, doc_prob))

            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                new_item['claim'] = ' '.join(easy_tokenize(item['claim']))

            new_item['evid'] = evidence_text_list_with_prob

            new_item['verifiable'] = item['verifiable']
            new_item['label'] = item['label']

            # print("C:", new_item['claim'])
            # print("E:", new_item['evid'])
            # print("L:", new_item['label'])
            # print()
            sampled_data_list.append(new_item)

    cursor.close()

    print(count)
    return sampled_data_list


def adv_simi_sample_with_prob_v1_1(input_file, additional_file, prob_dict_file, tokenized=False):
    cursor = fever_db.get_cursor()
    d_list = load_data(input_file)

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    sampled_data_list = []
    count = 0

    for item in tqdm(d_list):
        # e_list = check_sentences.check_and_clean_evidence(item)
        sampled_e_list, flags = sample_additional_data_for_item_v1_1(item, additional_data_dict)
        # print(flags)
        for i, (sampled_evidence, flag) in enumerate(zip(sampled_e_list, flags)):
            # Do not copy, might change in the future for error analysis
            # new_item = copy.deepcopy(item)
            new_item = dict()
            # print(new_item['claim'])
            # print(e_list)
            # print(sampled_evidence)
            # print(flag)
            evidence_text_list = evidence_list_to_text_list(
                cursor, sampled_evidence,
                contain_head=True, id_tokenized=tokenized)

            evidences = sorted(sampled_evidence, key=lambda x: (x[0], x[1]))
            item_id = int(item['id'])

            evidence_text_list_with_prob = []
            for text, (doc_id, ln) in zip(evidence_text_list, evidences):
                ssid = (int(item_id), doc_id, int(ln))
                if ssid not in prob_dict_file:
                    count += 1
                    print("Some sentence pair don't have 'prob'.")
                    prob = 0.5
                else:
                    prob = prob_dict_file[ssid]['prob']
                    assert item['claim'] == prob_dict_file[ssid]['claim']

                evidence_text_list_with_prob.append((text, prob))

            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                new_item['claim'] = ' '.join(easy_tokenize(item['claim']))

            new_item['evid'] = evidence_text_list_with_prob

            new_item['verifiable'] = item['verifiable']
            new_item['label'] = item['label']

            # print("C:", new_item['claim'])
            # print("E:", new_item['evid'])
            # print("L:", new_item['label'])
            # print()
            sampled_data_list.append(new_item)

    cursor.close()

    print(count)
    return sampled_data_list


def evidence_list_to_text_list(cursor, evidences, contain_head=True, id_tokenized=False):
    # One evidence one text and len(evidences) == len(text_list)
    current_evidence_text_list = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))

    cur_head = 'DO NOT INCLUDE THIS FLAG'

    for doc_id, line_num in evidences:

        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

        cur_text = ""

        if contain_head and cur_head != doc_id:
            cur_head = doc_id

            if not id_tokenized:
                doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            else:
                t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if line_num != 0:
                cur_text = f"{t_doc_id_natural_format} <t> "

        # Important change move one line below: July 16
        # current_evidence_text.append(e_text)
        cur_text = cur_text + e_text

        current_evidence_text_list.append(cur_text)

    assert len(evidences) == len(current_evidence_text_list)
    return current_evidence_text_list


def select_sent_with_prob_for_eval(input_file, additional_file, prob_dict_file, tokenized=False, pipeline=False):
    """
    This method select sentences with upstream sentence retrieval.

    :param input_file: This should be the file with 5 sentences selected.
    :return:
    """
    cursor = fever_db.get_cursor()

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    d_list = load_data(input_file)

    for item in tqdm(d_list):
        e_list = additional_data_dict[item['id']]['predicted_sentids']
        if not pipeline:
            assert additional_data_dict[item['id']]['label'] == item['label']
            assert additional_data_dict[item['id']]['verifiable'] == item['verifiable']
        assert additional_data_dict[item['id']]['id'] == item['id']

        pred_evidence_list = []
        for i, cur_e in enumerate(e_list):
            doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
            ln = int(cur_e.split(c_scorer.SENT_LINE)[1])  # Important changes Bugs: July 21
            pred_evidence_list.append((doc_id, ln))

        pred_evidence = check_sentences.Evidences(pred_evidence_list)

        evidence_text_list = evidence_list_to_text_list(cursor, pred_evidence,
                                                        contain_head=True, id_tokenized=tokenized)

        evidences = sorted(pred_evidence, key=lambda x: (x[0], x[1]))
        item_id = int(item['id'])

        evidence_text_list_with_prob = []
        for text, (doc_id, ln) in zip(evidence_text_list, evidences):
            ssid = (item_id, doc_id, int(ln))
            if ssid not in prob_dict_file:
                print("Some sentence pair don't have 'prob'.")
                prob = 0.5
            else:
                prob = prob_dict_file[ssid]['prob']
                assert item['claim'] == prob_dict_file[ssid]['claim']

            evidence_text_list_with_prob.append((text, prob))

        if tokenized:
            pass
        else:
            item['claim'] = ' '.join(easy_tokenize(item['claim']))

        item['evid'] = evidence_text_list_with_prob
        item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
        item['predicted_sentids'] = e_list
        # This change need to be saved.
        # item['predicted_label'] = additional_data_dict[item['id']]['label']

    return d_list


def select_sent_with_prob_for_eval_list(input_file, additional_file, prob_dict_file, tokenized=False, pipeline=False,
                                        is_demo=False):
    """
    This method select sentences with upstream sentence retrieval.

    :param input_file: This should be the file with 5 sentences selected.
    :return:
    """
    cursor = fever_db.get_cursor()

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    d_list = input_file

    for item in tqdm(d_list):
        e_list = additional_data_dict[item['id']]['predicted_sentids']
        if not pipeline:
            assert additional_data_dict[item['id']]['label'] == item['label']
            assert additional_data_dict[item['id']]['verifiable'] == item['verifiable']
        assert additional_data_dict[item['id']]['id'] == item['id']

        pred_evidence_list = []
        for i, cur_e in enumerate(e_list):
            doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
            ln = int(cur_e.split(c_scorer.SENT_LINE)[1])  # Important changes Bugs: July 21
            pred_evidence_list.append((doc_id, ln))

        pred_evidence = check_sentences.Evidences(pred_evidence_list)

        evidence_text_list = evidence_list_to_text_list(cursor, pred_evidence,
                                                        contain_head=True, id_tokenized=tokenized)

        evidences = sorted(pred_evidence, key=lambda x: (x[0], x[1]))
        item_id = int(item['id'])

        evidence_text_list_with_prob = []
        for text, (doc_id, ln) in zip(evidence_text_list, evidences):
            ssid = (item_id, doc_id, int(ln))
            if ssid not in prob_dict_file:
                print("Some sentence pair don't have 'prob'.")
                prob = 0.5
            else:
                prob = prob_dict_file[ssid]['prob']
                assert item['claim'] == prob_dict_file[ssid]['claim']

            evidence_text_list_with_prob.append((text, prob))

        if tokenized:
            pass
        else:
            item['claim'] = ' '.join(easy_tokenize(item['claim']))

        item['evid'] = evidence_text_list_with_prob
        item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
        item['predicted_sentids'] = e_list
        # This change need to be saved.
        # item['predicted_label'] = additional_data_dict[item['id']]['label']

    return d_list


# This function is added on top of the previous function for feeding document relatedness score to NLI models.
def select_sent_with_prob_doc_for_eval(input_file, additional_file,
                                       prob_dict_file,
                                       prob_doc_dict_file,
                                       tokenized=False, pipeline=False):
    """
    This method select sentences with upstream sentence retrieval.

    :param input_file: This should be the file with 5 sentences selected.
    :return:
    """
    cursor = fever_db.get_cursor()

    if isinstance(additional_file, list):
        additional_d_list = additional_file
    else:
        additional_d_list = load_data(additional_file)
    additional_data_dict = dict()

    for add_item in additional_d_list:
        additional_data_dict[add_item['id']] = add_item

    d_list = load_data(input_file)

    for item in tqdm(d_list):
        e_list = additional_data_dict[item['id']]['predicted_sentids']
        if not pipeline:
            assert additional_data_dict[item['id']]['label'] == item['label']
            assert additional_data_dict[item['id']]['verifiable'] == item['verifiable']
        assert additional_data_dict[item['id']]['id'] == item['id']

        pred_evidence_list = []
        for i, cur_e in enumerate(e_list):
            doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
            ln = int(cur_e.split(c_scorer.SENT_LINE)[1])  # Important changes Bugs: July 21
            pred_evidence_list.append((doc_id, ln))

        pred_evidence = check_sentences.Evidences(pred_evidence_list)

        evidence_text_list = evidence_list_to_text_list(cursor, pred_evidence,
                                                        contain_head=True, id_tokenized=tokenized)

        evidences = sorted(pred_evidence, key=lambda x: (x[0], x[1]))
        item_id = int(item['id'])

        evidence_text_list_with_prob = []
        for text, (doc_id, ln) in zip(evidence_text_list, evidences):
            ssid = (item_id, doc_id, int(ln))
            if ssid not in prob_dict_file:
                print("Some sentence pair don't have 'prob'.")
                prob = 0.5
            else:
                prob = prob_dict_file[ssid]['prob']
                assert item['claim'] == prob_dict_file[ssid]['claim']

            dssid = (item_id, doc_id)
            if dssid not in prob_doc_dict_file:
                # print("Some sentence pair don't have 'prob'.")
                doc_prob = 0.99
            else:
                doc_prob = prob_doc_dict_file[dssid]['prob']

                assert item['claim'] == prob_doc_dict_file[dssid]['claim']

            evidence_text_list_with_prob.append((text, prob, doc_prob))

        if tokenized:
            pass
        else:
            item['claim'] = ' '.join(easy_tokenize(item['claim']))

        item['evid'] = evidence_text_list_with_prob
        item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
        item['predicted_sentids'] = e_list
        # This change need to be saved.
        # item['predicted_label'] = additional_data_dict[item['id']]['label']

    return d_list


def get_adv_sampled_data(org_data_file, full_sent_list, threshold_prob=0.5, top_n=8):
    data_with_candidate_sample_list = \
        threshold_sampler(org_data_file, full_sent_list, threshold_prob, top_n=top_n)

    sampled_data = adv_sample_v1_0(org_data_file, data_with_candidate_sample_list, tokenized=True)

    return sampled_data


def format_printing(item):
    print("-" * 50)
    print("Claim:", item['claim'])
    print("Evidence:", item['evid'])
    # print("Pred Label:", item['predicted_label'])
    # print("Pred Evid:", item['predicted_evidence'])
    # print("Pred Evid F:", item['predicted_sentids'])
    # print("Label:", item['label'])
    # print("Evid:", item['evidence'])
    print("-" * 50)


if __name__ == '__main__':
    # sampled_data = get_adv_sampled_data(config.T_FEVER_DEV_JSONL,
    #                                     common.load_jsonl(
    #                                         config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl"),
    #                                     threshold_prob=0.35,
    #                                     top_n=8)
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
    #                                       prob_threshold=0.0, top_n=20)

    upstream_dev_list = threshold_sampler(config.T_FEVER_TRAIN_JSONL, dev_upstream_sent_list,
                                          prob_threshold=0.0, top_n=20)

    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    # train_sent_id_dict = set()
    print(len(selection_dict))

    # training_list = common.load_jsonl(config.T_FEVER_TRAIN_JSONL)
    # complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list, selection_dict, tokenized=True)
    # complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list, selection_dict, tokenized=True)

    complete_upstream_dev_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_TRAIN_JSONL, upstream_dev_list,
                                                                selection_dict,
                                                                tokenized=True)
    #
    # count = Counter()
    # length_list = []
    # for item in complete_upstream_dev_data:
    #     print(item)
    # length_list.extend([len(item['evid'].split(' '))])

    # count.update(length_list)
    # print(count.most_common())
    # print(sorted(list(count.most_common()), key=lambda x: -x[0]))
    # print(np.max(length_list))
    # print(np.mean(length_list))
    # print(np.std(length_list))
    #
    # for item in complete_upstream_dev_data[:5]:
    #     format_printing(item)

    # 785
    # 79.13041644297876
    # 43.75476065765309
