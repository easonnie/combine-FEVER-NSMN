#!/usr/bin/env python
"""Google doc retrieval!
"""
import requests
import re
from copy import copy

import config
from chaonan_src._utils.spcl import spcl

__all__ = ['GoogleQuerier']
__author__ = ['chaonan99']


class GoogleQuerier(object):
    """GoogleQuerier requires a built keyword processor. Also, item need
    to have claim tokens extracted.
    """
    def __init__(self, keyword_processor):
        self.kp = keyword_processor

    @classmethod
    def google_it(cls, keyword):
        quest_str = keyword.lower().replace(' ', '+')
        page = requests.get(f"https://www.google.com/search?q={quest_str}+wiki")
        # from IPython import embed; embed();
        matched_docid = re.search(r'(?<=https://en\.wikipedia\.org/wiki/)'\
                                  r'.+?&amp',
                                  page.text)
        return None if matched_docid is None else matched_docid[0][:-4]

    def get_keywords(self, claim):
        keyword_list = self.kp.extract_keywords(claim, span_info=True)
        finded_keywords = {claim[it[1]:it[2]]:it[0] for it in keyword_list}
        return finded_keywords

    def get_google_docid(self, item):
        finded_keywords = self.get_keywords(item['processed_claim'])
        google_docid = []
        quest_keywords = copy(finded_keywords)
        for keyword in finded_keywords:
            quest_keywords.remove(keyword)
            quest_keywords.insert(0, keyword)
            matched_docid = self.google_it(' '.join(quest_keywords))
            if matched_docid is not None:
                google_docid.append(matched_docid)
        item['google_docids'] = list(set(google_docid))


def main():
    # from chaonan_src._utils.doc_utils import read_jsonl
    pass
    # item_rb = ItemRuleBuilder()
    # d_list = read_jsonl(config.FEVER_TRAIN_JSONL)
    # test_google_doc_retrieval(d_list[:100], item_rb)
    # from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()
