# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""Document retrieval experiments for `FEVER <http://fever.ai/>`_

/***
 *      ┌─┐       ┌─┐ + +
 *   ┌──┘ ┴───────┘ ┴──┐++
 *   │                 │
 *   │       ───       │++ + + +
 *   ███████───███████ │+
 *   │                 │+
 *   │       ─┴─       │
 *   │                 │
 *   └───┐         ┌───┘
 *       │         │
 *       │         │   + +
 *       │         │
 *       │         └──────────────┐
 *       │                        │
 *       │                        ├─┐
 *       │                        ┌─┘
 *       │                        │
 *       └─┐  ┐  ┌───────┬──┐  ┌──┘  + + + +
 *         │ ─┤ ─┤       │ ─┤ ─┤
 *         └──┴──┘       └──┴──┘  + + + +
 *             God bless
 *               No BUG
 */
"""

import os
import json
from time import time
from copy import copy
from collections import Counter

import inflection
import numpy as np

import config
from sentence_retrieval.sent_tfidf import OnlineTfidfDocRanker
from drqa_yixin.tokenizers import CoreNLPTokenizer, set_default
from utils.c_scorer import fever_score, check_doc_id_correct
from chaonan_src._doc_retrieval.item_rules import ItemRuleBuilder, \
                                                  ItemRuleBuilderLower, \
                                                  ItemRuleBuilderRawID
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral
from chaonan_src._utils.spcl import spcl
from chaonan_src._utils.doc_utils import FEVERScorer, read_jsonl


__all__ = ['DocRetrievalExperiment']
__author__ = ['chaonan99', 'yixin1']


class DocRetrievalExperiment(object):
    """DocRetrievalExperiment
    """
    def __init__(self, item_rb=None):
        # self.item_rb = ItemRuleBuilder() if item_rb is None else item_rb
        # self.item_rb = ItemRuleBuilderLower() if item_rb is None else item_rb
        self.item_rb = ItemRuleBuilderRawID() if item_rb is None else item_rb

    def sample_answer_with_priority(self, d_list, top_k=5):
        for i, item in enumerate(spcl(d_list)):
            self.item_rb.rules(item)
            item['predicted_docids'] = \
                list(set([k for k, v \
                            in sorted(item['prioritized_docids'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))

    @classmethod
    def print_eval(cls, d_list):
        print(f"Acc {FEVERScorer.doc_loose_acc(d_list)}")
        print(f"F1 {FEVERScorer.doc_f1(d_list)}")
        print(f"Ave len {FEVERScorer.average_docid_number(d_list)}")
        print(cls.count_evidence(d_list))

    @classmethod
    def extract_failure(cls, d_list):
        correct = [check_doc_id_correct(item) for item in d_list]
        return np.array(d_list)[~np.array(correct)].tolist()

    @classmethod
    def sample_item(cls, d_list):
        return np.random.choice(d_list)

    @classmethod
    def dump_results(cls, d_list, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, 'w') as f:
            f.write('\n'.join(map(json.dumps, d_list)))

    @classmethod
    def count_evidence(cls, d_list):
        return Counter(list(map(lambda x: len(x.get('predicted_docids')),
                                d_list)))


class DocRetrievalExperimentSpiral(DocRetrievalExperiment):
    """docstring for DocRetrievalExperimentSpiral"""
    def __init__(self, item_rb=None):
        self.item_rb = ItemRuleBuilderSpiral() if item_rb is None else item_rb

    def sample_answer_with_priority(self, d_list, top_k=5):
        for i, item in enumerate(spcl(d_list)):
            self.item_rb.rules(item)
            item['predicted_docids_origin'] = \
                list(set([k for k, v \
                            in sorted(item['prioritized_docids'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))
            item['predicted_docids_aside'] = \
                list(set([k for k, v \
                            in sorted(item['prioritized_docids_aside'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))
            item['predicted_docids'] = item['predicted_docids_origin'] + \
                                       item['predicted_docids_aside']


class DocRetrievalExperimentTwoStep(DocRetrievalExperimentSpiral):
    """docstring for DocRetrievalExperimentTwoStep"""
    def __init__(self, item_rb=None):
        super(DocRetrievalExperimentTwoStep, self).__init__(item_rb)

    def sample_answer_with_priority(self, d_list, top_k=5):
        for i, item in enumerate(spcl(d_list)):
            self.item_rb.first_only_rules(item)
            item['predicted_docids'] = \
                list(set([k for k, v \
                            in sorted(item['prioritized_docids'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))

    def feed_sent_file(self, path):
        """Use absolute path, and it should appear in my repo"""
        self.item_rb.feed_sent_score_result(path)

    def find_sent_link_with_priority(self, d_list, top_k=5, predict=False):
        for item in spcl(d_list):
            self.item_rb.second_only_rules(item)
            pids = [it[0] for it in item['prioritized_docids']]
            item['prioritized_docids_aside'] = \
                [it for it in item['prioritized_docids_aside']\
                    if it[0] not in pids]
            if predict:
                porg = \
                    set([k for k, v \
                           in sorted(item['prioritized_docids'],
                                     key=lambda x: (-x[1], x[0]))][:top_k])
                paside = \
                    set([k for k, v \
                            in sorted(item['prioritized_docids_aside'],
                                      key=lambda x: (-x[1], x[0]))][:top_k])
                item['predicted_docids'] = list(porg | paside)
                item['predicted_docids_origin'] = list(porg)
                item['predicted_docids_aside'] = list(paside)


def main():
    doc_exp = DocRetrievalExperimentTwoStep()

    d_list = read_jsonl(config.FEVER_DEV_JSONL)
    # d_list = read_jsonl(config.FEVER_TRAIN_JSONL)

    doc_exp.sample_answer_with_priority(d_list)
    doc_exp.print_eval(d_list)
    doc_exp.feed_sent_file("../../results/"
        "sent_retri_nn/2018_07_17_16-34-19_r/dev_scale(0.1).jsonl")
    doc_exp.find_sent_link_with_priority(d_list, predict=True)
    doc_exp.print_eval(d_list)

    # path = "../../results/doc_retri/docretri.spiral.aside/train.jsonl"
    # doc_exp.dump_results(d_list, path)

    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()