import json
from collections import Counter
import numpy as np

from sentence_retrieval.sent_tfidf import OnlineTfidfDocRanker
from utils import fever_db, check_sentences, text_clean
import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from tqdm import tqdm
from utils import c_scorer
import math
import utils
from collections import namedtuple
from pathlib import Path

from utils import common

path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
print(path_stanford_corenlp_full_2017_06_09)

drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
tok = CoreNLPTokenizer(annotators=['pos', 'lemma'])


def easy_tokenize(text):
    return tok.tokenize(text_clean.normalize(text)).words()


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def utest_for_ground_truth(d_list):
    nei_c = 0
    support_c = 0
    refute_c = 0
    for item in tqdm(d_list):
        e_list = check_sentences.check_and_clean_evidence(item)
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

        if item["label"] == 'NOT ENOUGH INFO':
            nei_c += 1
        elif item["label"] == 'SUPPORTS':
            support_c += 1
        elif item["label"] == 'REFUTES':
            refute_c += 1

    print(support_c, refute_c, nei_c)
        # if len(evidence_sent_id) >= 2:
        #     print(evidence_sent_id)


def utest_score_ground_truth():
    d_list = load_data(config.FEVER_DEV_JSONL)
    utest_for_ground_truth(d_list)

    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    print(c_scorer.fever_score(d_list, d_list, mode=eval_mode, verbose=False))


def utest_check_sentence_lines():
    sent_number_coutner = Counter()
    number_list = []
    db_cursor = fever_db.get_cursor()
    # d_list = load_data("/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/dev.jsonl")
    d_list = load_data("/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/train.jsonl")
    for item in tqdm(d_list):
        p_docids = item['predicted_docids']
        current_sent_list = []
        for doc_id in p_docids:
            r_list = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id)
            current_sent_list.extend(r_list)

        sent_number_coutner.update([len(current_sent_list)])
        number_list.append(len(current_sent_list))
        # print(current_sent_list)

    print(len(number_list))
    print('Mean:', np.mean(number_list))
    print('Max:', np.max(number_list))
    print('Min:', np.min(number_list))
    print('Std:', np.std(number_list))
    print(sent_number_coutner)

        # print(p_docids)


def save_intermidiate_results(d_list, out_filename: Path, last_loaded_path=None):
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(exist_ok=False)

    with open(out_filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')

    if last_loaded_path is not None:
        with open(out_filename.parent / "log_info.txt", encoding='utf-8', mode='a') as out_f:
            out_f.write(last_loaded_path)


def if_idf_select_sentence():
    db_cursor = fever_db.get_cursor()
    loaded_path = "/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/dev.jsonl"
    d_list = load_data(loaded_path)
    # d_list = load_data("/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/train.jsonl")

    for item in tqdm(d_list):
        # print()
        p_docids = item['predicted_docids']
        cleaned_claim = ' '.join(easy_tokenize(item['claim']))
        # print(cleaned_claim)

        current_sent_list = []
        current_id_list = []
        for doc_id in p_docids:
            r_list, id_list = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id)
            current_sent_list.extend(r_list)
            current_id_list.extend(id_list)

        Args = namedtuple('Args', 'ngram hash_size num_workers')

        args = Args(2, int(8192), 4)

        ranker = OnlineTfidfDocRanker(args, args.hash_size, args.ngram,
                                      current_sent_list)

        selected_index, selected_score = ranker.closest_docs(cleaned_claim, k=5)

        selected_sent_id = []
        for ind in selected_index:
            curent_selected = current_id_list[ind]
            doc_id, ln = curent_selected.split('(-.-)')
            # ln = int(ln)
            # selected_sent_id.append([doc_id, ln])
            selected_sent_id.append(doc_id + c_scorer.SENT_LINE + ln)

        item['predicted_sentids'] = selected_sent_id

    eval_mode = {'check_sent_id_correct': True, 'standard': False}
    print(c_scorer.fever_score(d_list, d_list, mode=eval_mode, verbose=False))

    out_fname = config.RESULT_PATH / "sent_retri" / f"{utils.get_current_time_str()}_r" / "dev.jsonl"
    save_intermidiate_results(d_list, out_filename=out_fname, last_loaded_path=loaded_path)


def script(in_path, out_path):
    print(in_path)
    print(out_path)
    db_cursor = fever_db.get_cursor()

    if isinstance(in_path, str):
        loaded_path = in_path
        d_list = load_data(loaded_path)
    elif isinstance(in_path, list):
        d_list = in_path
    else:
        print("Error input format")
        d_list = None
        exit(-1)

    # d_list = load_data("/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/train.jsonl")
    res_list = []

    for item in tqdm(d_list):
        # print()
        p_docids = item['predicted_docids']
        # cleaned_claim = ' '.join(easy_tokenize(item['claim']))
        # print(cleaned_claim)
        cleaned_claim = ' '.join(item['claim'].split(' '))

        current_sent_list = []
        current_id_list = []
        for doc_id in p_docids:
            r_list, id_list = fever_db.get_all_sent_by_doc_id(db_cursor, doc_id)
            current_sent_list.extend(r_list)
            current_id_list.extend(id_list)

        Args = namedtuple('Args', 'ngram hash_size num_workers')

        args = Args(2, int(8192), 4)

        ranker = OnlineTfidfDocRanker(args, args.hash_size, args.ngram,
                                      current_sent_list)

        selected_index, selected_score = ranker.closest_docs(cleaned_claim, k=5)

        selected_sent_id = []
        for ind in selected_index:
            curent_selected = current_id_list[ind]
            doc_id, ln = curent_selected.split('(-.-)')
            # ln = int(ln)
            # selected_sent_id.append([doc_id, ln])
            selected_sent_id.append(doc_id + c_scorer.SENT_LINE + ln)

        item['predicted_sentids'] = selected_sent_id

        res_list.append(item)
        # break

    eval_mode = {'check_sent_id_correct': True, 'standard': False}
    print(c_scorer.fever_score(res_list, res_list, mode=eval_mode, verbose=False))

    # out_fname = config.RESULT_PATH / "sent_retri" / f"{utils.get_current_time_str()}_r" / "dev.jsonl"
    out_fname = Path(out_path)
    save_intermidiate_results(res_list, out_filename=out_fname, last_loaded_path=None)


def check_acc(in_path):
    d_list = load_data(in_path)
    eval_mode = {'check_sent_id_correct': True, 'standard': False}
    print(c_scorer.fever_score(d_list, d_list, mode=eval_mode, verbose=False))


if __name__ == '__main__':
    # check_acc("/Users/Eason/RA/FunEver/results/sent_retri/2018_07_05_17:17:50_r/train.jsonl")

    # utest_check_sentence_lines()
    # check_acc(in_path)
    # d_list = load_data(in_path)
    # if_idf_select_sentence()

    # script(
    #     in_path='/Users/Eason/RA/FunEver/results/doc_retri/docretri.pageview/dev.jsonl',
    #     out_path='/Users/Eason/RA/FunEver/results/sent_retri/docretri.pageview.tf_idf/dev.jsonl'
    # )

    # script(
    #     in_path=common.load_jsonl('/Users/Eason/RA/FunEver/results/doc_retri/docretri.pageview/dev.jsonl'),
    #     out_path='/Users/Eason/RA/FunEver/results/sent_retri/docretri.pageview.tf_idf/dev_0.jsonl'
    # )

    doc_num = 10

    dev_doc_file = config.SRC_ROOT / "analysis/analysis_data/sentence_selection/upstream_nn_doc_retri.jsonl"
    dev_doc_list = common.load_jsonl(dev_doc_file)

    for item in dev_doc_list:
        item['predicted_docids'] = item['predicted_docids'][:doc_num]

    script(
        in_path=dev_doc_list,
        out_path=str(config.SRC_ROOT / f"analysis/analysis_data/sentence_selection/tf_idf_sent_selection_docn({doc_num}).jsonl")
    )

    # 16783 0.8392339233923393 document = 5 upstream
    # 16752 0.8376837683768377 document = 10 upstream


    # utest_score_ground_truth()
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

