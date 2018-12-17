#!/usr/bin/env python
# -*- coding: utf-8 -*-


from copy import deepcopy

from chaonan_src._utils.doc_utils import read_jsonl
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral
from chaonan_src._doc_retrieval.item_rules_raw_pageview import \
    ItemRuleRawPageview
from chaonan_src.doc_retrieval_experiment import DocRetrievalExperiment
import config


__author__ = ['chaonan99']


class ItemRuleBuilderExp(ItemRuleBuilderSpiral):
    """Specify your own rule set!"""
    @property
    def rules(self):
        return lambda x: self.initialize_item(x)\
                             .exact_match_rule()\
                             .docid_based_rule()\
                             .singularize_rule()\
                             .eliminate_articles_rule()


def change_name(d_list_ori, d_list_pageview):
    for it_ori, it_pageview in zip(d_list_ori, d_list_pageview):
        it_pageview['docid_pageviews'] = \
            deepcopy(it_pageview['prioritized_docids'])
        it_pageview['prioritized_docids'] = it_ori['prioritized_docids']


def pageview_threshold():
    pass


def main():
    import os
    from chaonan_src._config import old_result_path
    from chaonan_src._utils.doc_utils import read_jsonl
    from chaonan_src._utils.spcl import spcl

    # pageview_path = os.path.join(config.RESULT_PATH,
    #                              'doc_retri/docretri.rawpageview/train.jsonl')
    pageview_path = config.RESULT_PATH / \
                    'doc_retri/docretri.rawpageview/dev.jsonl'
    # ori_path = os.path.join(old_result_path,
    #                         'doc_retri/docretri.pageview/dev.jsonl')

    # d_list = read_jsonl(config.FEVER_DEV_JSONL)
    # item_rb_exp = ItemRuleRawPageview()
    # doc_exp = DocRetrievalExperiment(item_rb_exp)
    # doc_exp.sample_answer_with_priority(d_list)
    # doc_exp.dump_results(d_list, save_path)
    # doc_exp.print_eval(d_list)

    # d_list_ori = read_jsonl(ori_path)
    d_list = read_jsonl(pageview_path)

    # DocRetrievalExperiment.dump_results(d_list, pageview_path)
    # item_rb = ItemRuleRawPageview()

    top_k = 5
    for item in spcl(d_list):
        item['predicted_docids'] = \
                list(set([k for k, v \
                            in sorted(item['docid_pageviews'],
                                      key=lambda x: (-x[1], x[0]))][:top_k]))
    # DocRetrievalExperiment.dump_results(d_list, save_path)
    DocRetrievalExperiment.print_eval(d_list)
    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()