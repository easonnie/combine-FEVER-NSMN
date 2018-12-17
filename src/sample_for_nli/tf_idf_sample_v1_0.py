import json
import random
import copy

from utils import fever_db, check_sentences
import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from tqdm import tqdm
from utils import c_scorer, text_clean, common
from collections import Counter
import numpy as np


class DrQaTokenizer:
    def __init__(self):
        self.instance = None

    def create_instance(self):
        path_stanford_corenlp_full_2017_06_09 = \
            str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
        print("Load tokenizer:", path_stanford_corenlp_full_2017_06_09)
        drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
        _tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])
        self.instance = _tok

    def clean(self):
        self.instance = None


tok = DrQaTokenizer()
tok.clean()

random.seed = 12


def easy_tokenize(text):
    if tok.instance is None:
        tok.create_instance()
    return tok.tokenize(text_clean.normalize(text)).words()


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

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


def convert_to_normalized_format(cursor, e_list, contain_head=True):
    r_list = []
    for evidences in e_list:
        current_evidence = []
        cur_head = 'DO NOT INCLUDE THIS FLAG'
        # if len(evidences) >= 2:
        #     print("!!!")

        # This is important sorting of all evidences.
        evidences = sorted(evidences, key=lambda x: (x[0], x[1]))
        # print(evidences)

        for doc_id, line_num in evidences:

            _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)

            if contain_head and cur_head != doc_id:
                cur_head = doc_id

                doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
                t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))

                if line_num != 0:
                    current_evidence.append(f"{t_doc_id_natural_format} .")

            # print(e_text)
            current_evidence.append(e_text)
        # print(current_evidence)
        r_list.append(' '.join(current_evidence))

    return r_list


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


def sample_v1_0(input_file, additional_file, tokenized=False):
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

    return sampled_data_list


def select_sent_for_eval(input_file, additional_file, tokenized=False):
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
        assert additional_data_dict[item['id']]['label'] == item['label']
        assert additional_data_dict[item['id']]['id'] == item['id']
        assert additional_data_dict[item['id']]['verifiable'] == item['verifiable']

        pred_evidence_list = []
        for i, cur_e in enumerate(e_list):
            doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
            ln = int(cur_e.split(c_scorer.SENT_LINE)[1])    # Important changes Bugs: July 21
            pred_evidence_list.append((doc_id, ln))

        pred_evidence = check_sentences.Evidences(pred_evidence_list)

        evidence_text = evidence_list_to_text(cursor, pred_evidence,
                                              contain_head=True, id_tokenized=tokenized)

        if tokenized:
            pass
        else:
            item['claim'] = ' '.join(easy_tokenize(item['claim']))

        item['evid'] = evidence_text
        item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
        item['predicted_sentids'] = e_list
        # This change need to be saved.
        # item['predicted_label'] = additional_data_dict[item['id']]['label']

    return d_list

# This method might be useful in the future.
# def select_ground_truth_re_sent_for_eval(input_file, tokenized=False):
#     """
#     This method select sentences with ground truth sentences.
#
#     :param input_file: This should be the file with 5 sentences selected.
#     :return:
#     """
#     cursor = fever_db.get_cursor()
#
#     # additional_d_list = load_data(additional_file)
#     # additional_data_dict = dict()
#     #
#     # for add_item in additional_d_list:
#     #     additional_data_dict[add_item['id']] = add_item
#
#     d_list = load_data(input_file)
#
#     for item in tqdm(d_list):
#         # e_list = additional_data_dict[item['id']]['predicted_sentids']
#         # assert additional_data_dict[item['id']]['label'] == item['label']
#         # assert additional_data_dict[item['id']]['id'] == item['id']
#         # assert additional_data_dict[item['id']]['verifiable'] == item['verifiable']
#
#         pred_evidence_list = []
#         for i, cur_e in enumerate(e_list):
#             doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
#             ln = cur_e.split(c_scorer.SENT_LINE)[1]
#             pred_evidence_list.append((doc_id, ln))
#
#         pred_evidence = check_sentences.Evidences(pred_evidence_list)
#
#         evidence_text = evidence_list_to_text(cursor, pred_evidence,
#                                               contain_head=True, id_tokenized=tokenized)
#
#         if tokenized:
#             pass
#         else:
#             item['claim'] = ' '.join(easy_tokenize(item['claim']))
#
#         item['evid'] = evidence_text
#         item['predicted_evidence'] = convert_evidence2scoring_format(e_list)
#         item['predicted_sentids'] = e_list
#
#     cursor.close()
#
#     return d_list


def convert_evidence2scoring_format(predicted_sentids):
    e_list = predicted_sentids
    pred_evidence_list = []
    for i, cur_e in enumerate(e_list):
        doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
        ln = cur_e.split(c_scorer.SENT_LINE)[1]
        pred_evidence_list.append([doc_id, int(ln)])
    return pred_evidence_list


def save_jsonl(d_list, filename):
    with open(filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    # input_file = config.FEVER_DEV_JSONL
    # additional_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    # sampled_d_list = sample_v1_0(input_file, additional_file)
    # print(len(sam))

    # input_file = config.T_FEVER_TRAIN_JSONL
    # additional_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/train.jsonl"
    # sampled_d_list = sample_v1_0(input_file, additional_file, tokenized=True)
    # print(len(sampled_d_list))
    # save_jsonl(sampled_d_list, "/Users/Eason/RA/FunEver/results/tmp/utest_sampled_data/train_0.jsonl")

    # print(len(sampled_d_list))
    # save_jsonl(sampled_d_list, "/Users/Eason/RA/FunEver/results/tmp/utest_sampled_data/t_dev_1.jsonl")
    input_file = config.T_FEVER_DEV_JSONL
    # input_file = config.T_FEVER_TRAIN_JSONL
    # input_file = config.FEVER_DEV_JSONL
    additional_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    # additional_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/train.jsonl"
    # sampled_d_list = sample_v1_0(input_file, additional_file, tokenized=True)
    sampled_d_list = select_sent_for_eval(input_file, additional_file, tokenized=True)

    # for item in sampled_d_list:
    #     print(item[''])

    count = Counter()
    length_list = []
    for item in sampled_d_list:
        length_list.extend([len(item['evid'].split(' '))])

    count.update(length_list)
    print(count.most_common())
    print(sorted(list(count.most_common()), key=lambda x: -x[0]))
    print(np.max(length_list))
    print(np.mean(length_list))
    print(np.std(length_list))

    # print(count.max())

    # print(len(sampled_d_list))
    # save_jsonl(sampled_d_list, "/Users/Eason/RA/FunEver/results/tmp/utest_sampled_data/t_dev_1.jsonl")