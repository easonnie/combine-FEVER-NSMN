import json
import random
import copy

from utils import fever_db, check_sentences
import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from tqdm import tqdm
from utils import c_scorer, text_clean
from utils import common

path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
print(path_stanford_corenlp_full_2017_06_09)

drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])

random.seed = 12


def easy_tokenize(text):
    return tok.tokenize(text_clean.normalize(text)).words()


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def sample_for_verifiable(cursor, e_list, contain_head=True):
    r_list = []
    for evidences in e_list:
        current_evidence = []
        cur_head = 'DO NOT INCLUDE THIS FLAG'
        # if len(evidences) >= 2:
        #     print("!!!")
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


def sample_for_non_verifiable(cursor, contain_head=True):
    r_list = []
    pass
    # for evidences in e_list:
    #     current_evidence = []
    #     cur_head = 'DO NOT INCLUDE THIS FLAG'
    #
        # for doc_id, line_num in evidences:
        #     print(doc_id, line_num)
            # _, e_text, _ = fever_db.get_evidence(cursor, doc_id, line_num)
            #
            # if contain_head and cur_head != doc_id:
            #     cur_head = doc_id
            #
            #     doc_id_natural_format = fever_db.convert_brc(doc_id).replace('_', ' ')
            #     t_doc_id_natural_format = ' '.join(easy_tokenize(doc_id_natural_format))
            #
            #     if line_num != 0:
            #         current_evidence.append(f"{t_doc_id_natural_format} .")
            #
            # current_evidence.append(e_text)
        #
        # print(current_evidence)
        # r_list.append(' '.join(current_evidence))
    #
    # return r_list



# global count
# count = 0


def sample_additional_data_for_item(item, additional_data_dictionary):
    #TODO check this code, very messy!!!
    res_sentids_list = []
    flags = []
    # count = 0
    # total = 0
    # print(count, total)

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
                flag = f"verifiable.eq.{len(new_evidences) - len(evidences)}"
                flags.append(flag)
                # count += 1
                # print("Oh")
                # print(evidences)
                # print(new_evidences)
                pass
            else:
                flag = "verifiable.eq.0"
                flags.append(flag)
                pass
                    # print("Yes")
            res_sentids_list.append(new_evidences)
            # total += 1
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

        # print(e_list)
        # print(new_evidences)

        if len(new_evidences) == 0:
            flag = f"verifiable.eq.0"
            flags.append(flag)
            pass
        else:
            flag = f"not_verifiable.eq.{len(new_evidences)}"
            flags.append(flag)

# print("Oh")
            # global count
            # count += 1
            # print(additional_data)

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


def utest_for_ground_truth(d_list):
    for item in tqdm(d_list):
        e_list = check_sentences.check_and_clean_evidence(item)
        print(e_list)
        evidence_sent_id = []
        gt_evidence = []
        if item['verifiable'] == "VERIFIABLE":
            for doc_id, ln in list(e_list)[0]:
                evidence_sent_id.append(doc_id + c_scorer.SENT_LINE + str(ln))
                gt_evidence.append([doc_id, ln])
        elif item['verifiable'] == "NOT VERIFIABLE":
            evidence_sent_id = []

        item["predicted_sentids"] = evidence_sent_id

        # item['predicted_evidence'] = []
        item['predicted_evidence'] = gt_evidence
        item['predicted_label'] = item["label"]
        # if len(evidence_sent_id) >= 2:
        #     print(evidence_sent_id)


def utest_score_ground_truth():
    d_list = load_data(config.FEVER_DEV_JSONL)
    utest_for_ground_truth(d_list)

    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    print(c_scorer.fever_score(d_list, d_list, mode=eval_mode, verbose=False))


def utest_for_sample():
    # cursor = fever_db.get_cursor()
    d_list = load_data(config.FEVER_DEV_JSONL)
    additional_d_list = load_data(config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl")
    additional_data_dict = dict()
    for item in additional_d_list:
        additional_data_dict[item['id']] = item

    for item in d_list:
        e_list = check_sentences.check_and_clean_evidence(item)
        # e_text_list = convert_to_normalized_format(cursor, e_list)
        r_list, flags = sample_additional_data_for_item(item, additional_data_dict)
        print(flags)
        # print(e_list)
        # print(e_text_list)

    # print(count)


if __name__ == '__main__':
    # utest_score_ground_truth()
    utest_for_sample()
    # d_list = load_data(config.DATA_ROOT / "fever/shared_task_dev.jsonl")
    # db_cursor = fever_db.get_cursor()
    #
    # contain_head = True
    #
    # for item in tqdm(d_list):
    #     e_list = check_sentences.check_and_clean_evidence(item)
    #     evidence_text_list = []
    #     if item['verifiable'] == "VERIFIABLE":
    #         evidence_text_list = sample_for_verifiable(db_cursor, e_list, contain_head=contain_head)
    #         print(evidence_text_list)
    #     elif item['verifiable'] == "NOT VERIFIABLE":
    #         pass

        # if len(evidence_text_list) >= 2:
        #     print("Claim:", item['claim'])
        #     for text in evidence_text_list:
                # print(text)

