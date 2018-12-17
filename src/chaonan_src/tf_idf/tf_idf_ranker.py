#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import math
from uuid import uuid4

import numpy as np

import config
from sentence_retrieval.sent_tfidf import OnlineTfidfDocRanker
from chaonan_src._utils.doc_utils import read_jsonl
from chaonan_src._utils.spcl import spcl
from chaonan_src.doc_retrieval_experiment import DocRetrievalExperiment
from utils.fever_db import get_all_sent_by_doc_id, get_cursor


__author__ = ['chaonan99']


def tf_idf_rank(args, top_k=5):
    dev_path = config.PRO_ROOT / \
               'results_old/doc_retri/docretri.basic.nopageview/dev.jsonl'

    cursor = get_cursor()
    d_list = read_jsonl(dev_path)

    d_list_test = d_list

    for i, item in enumerate(spcl(d_list_test)):
        all_sent = []
        all_ids = [it[0] for it in item['prioritized_docids']]

        try:

            for doc_id in all_ids:
                r_list, _ = get_all_sent_by_doc_id(cursor,
                                                   doc_id,
                                                   with_h_links=False)
                all_sent.append(' '.join(r_list))

            ranker = OnlineTfidfDocRanker(args,
                                          args.hash_size,
                                          args.ngram,
                                          all_sent)
        except Exception as e:
            if i - 1 >= 0:
                print(f'Early quit at {i-1} because of {e}')
                save_path = config.RESULT_PATH / \
                            'doc_retri/docretri.tfidfrank/' \
                            f'dev_quit_dump_{uuid4()}.json'
                DocRetrievalExperiment.dump_results(d_list_test[:i], save_path)
            raise e

        rank_ind, rank_score = \
            ranker.closest_docs(' '.join(item['claim_tokens']), k=100)
        id_score_dict = {docid: 0 for docid in all_ids}
        id_score_dict.update({all_ids[ri]: rs \
                              for ri, rs in zip(rank_ind, rank_score)})
        item['prioritized_docids'] = [(k, v) for k, v in id_score_dict.items()]
        item['predicted_docids'] = \
                list(set([k for k, v \
                            in sorted(item['prioritized_docids'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))

    save_path = config.RESULT_PATH / 'doc_retri/docretri.tfidfrank/dev.json'
    DocRetrievalExperiment.dump_results(d_list_test, save_path)


def eval():
    save_path = config.RESULT_PATH / 'doc_retri/docretri.tfidfrank/dev_toy.json'
    d_list = read_jsonl(save_path)
    DocRetrievalExperiment.print_eval(d_list)


def main(args):
    tf_idf_rank(args)
    # eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()
    main(args)