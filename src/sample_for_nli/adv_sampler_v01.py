import json
import random
import copy

# from nli.mesim_wn_v1_0 import get_actual_data
from sample_for_nli.tf_idf_sample_v1_0 import select_sent_for_eval
from sentence_retrieval.nn_postprocess_ablation import score_converter_scaled
from utils import fever_db, check_sentences
import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from tqdm import tqdm
from utils import c_scorer, text_clean, common
from collections import Counter
import numpy as np


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


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
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score']))
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids
        item['predicted_sentids'] = [sid for sid, _ in item['scored_sentids']][:top_n]
        # item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
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


def adv_sample_v1_0(input_file, additional_file, tokenized=False):
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
            evidence_text = evidence_list_to_text(cursor, sampled_evidence,
                                                  contain_head=True, id_tokenized=tokenized)

            new_item['id'] = str(item['id']) + '#' + str(i)

            if tokenized:
                new_item['claim'] = item['claim']
            else:
                new_item['claim'] = ' '.join(easy_tokenize(item['claim']))

            new_item['evid'] = evidence_text

            new_item['verifiable'] = item['verifiable']
            new_item['label'] = item['label']

            # print("C:", new_item['claim'])
            # print("E:", new_item['evid'])
            # print("L:", new_item['label'])
            # print()
            sampled_data_list.append(new_item)

    cursor.close()

    return sampled_data_list


def evidence_list_to_text(cursor, evidences, contain_head=True, id_tokenized=False):
    current_evidence_text = []
    evidences = sorted(evidences, key=lambda x: (x[0], x[1]))

    cur_head = 'DO NOT INCLUDE THIS FLAG'

    for doc_id, line_num in evidences:

        _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

        if contain_head and cur_head != doc_id:
            cur_head = doc_id

            if not id_tokenized:
                doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            else:
                t_doc_id_natural_format = common.doc_id_to_tokenized_text(doc_id)

            if line_num != 0:
                current_evidence_text.append(f"{t_doc_id_natural_format} <t>")

        # Important change move one line below: July 16
        current_evidence_text.append(e_text)

    # print(current_evidence_text)

    return ' '.join(current_evidence_text)


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
    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")

    upstream_dev_list = score_converter_scaled(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                               scale_prob=0.5,
                                               delete_prob=False)

    complete_upstream_dev_data = adv_sample_v1_0(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                 tokenized=True)

    count = Counter()
    length_list = []
    for item in complete_upstream_dev_data:
        length_list.extend([len(item['evid'].split(' '))])

    count.update(length_list)
    print(count.most_common())
    print(sorted(list(count.most_common()), key=lambda x: -x[0]))
    print(np.max(length_list))
    print(np.mean(length_list))
    print(np.std(length_list))

    for item in complete_upstream_dev_data[:5]:
        format_printing(item)

    # 785
    # 79.13041644297876
    # 43.75476065765309
