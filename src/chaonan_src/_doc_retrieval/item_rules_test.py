# -*- coding: utf-8 -*-
#!/usr/bin/env python


import pickle
import numpy as np

import config
from utils.text_clean import STOPWORDS
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral


__all__ = ['ItemRuleBuilderTest', 'ItemRuleBuilderNoBracket']
__author__ = ['chaonan99']


class ItemRuleBuilderTest(ItemRuleBuilderSpiral):
    """docstring for ItemRuleBuilderTest"""
    def __init__(self, tokenizer=None, keyword_processor=None):
        freq_dict_path = config.RESULT_PATH / 'chaonan99/freq_dict.pkl'
        self.freq_dict = pickle.load(open(freq_dict_path, 'rb'))
        super(ItemRuleBuilderTest, self).__init__(tokenizer,
                                                  keyword_processor)

    def _recursive_key_matcher(self, claim, th=5):
        keyword_list = self.keyword_processor.extract_keywords(claim,
                                                               span_info=True)
        finded_keys_dict = {claim[it[1]:it[2]]:it[0] for it in keyword_list}

        fkd = {}

        for value, start, end in keyword_list:
            key = claim[start:end]
            if key.lower() in STOPWORDS:
                continue

            ## First letter is case sensitive
            key_tokens = key.split(' ')
            if np.all(list(map(lambda x: self.freq_dict[x.lower()] > th,
                               key_tokens))):
                new_value = [v for v in value if v[0][0] == key[0]]
            else:
                new_value = list(value)

            if len(new_value) == 0:
                # print(value)
                ## Failed
                # key_tokens = key.split(' ')
                if len(key_tokens) > 2:
                    new_claim = ' '.join([claim[:start-1],
                                          *key_tokens[1:],
                                          claim[end+1:]])
                    return self._recursive_key_matcher(new_claim, th)
            else:
                fkd.update({key:new_value})

        return fkd


class ItemRuleBuilderNoBracket(ItemRuleBuilderTest):
    """docstring for ItemRuleBuilderNoBracket"""
    def __init__(self, tokenizer=None, keyword_processor=None):
        super(ItemRuleBuilderNoBracket, self).__init__(tokenizer,
                                                       keyword_processor)

    def _recursive_key_matcher(self, claim, th=2):
        keyword_list = self.keyword_processor.extract_keywords(claim,
                                                               span_info=True)
        finded_keys_dict = {claim[it[1]:it[2]]:it[0] for it in keyword_list}

        fkd = {}

        for value, start, end in keyword_list:
            key = claim[start:end]
            if key.lower() in STOPWORDS:
                continue

            ## First letter is case sensitive
            key_tokens = key.split(' ')
            if np.all(list(map(lambda x: self.freq_dict[x.lower()] > th,
                               key_tokens))):
                new_value = [v for v in value if v[0][0] == key[0]]
            else:
                new_value = [v for v in value if '-LRB-' not in v[0]] \
                               if (key[0] >= 'a' and key[0] <= 'z') \
                               else list(value)

            if len(new_value) == 0:
                # print(value)
                ## Failed
                # key_tokens = key.split(' ')
                if len(key_tokens) > 2:
                    new_claim = ' '.join([claim[:start-1],
                                          *key_tokens[1:],
                                          claim[end+1:]])
                    return self._recursive_key_matcher(new_claim, th)
            else:
                fkd.update({key:new_value})

        return fkd



def main():
    item_rb_test = ItemRuleBuilderTest()
    from IPython import embed; embed(); import os; os._exit(1)
    item = {'id':0, 'claim':'During diabetic ketoacidosis the body stores acidic ketone bodies .'}
    item_rb_test.initialize_item(item).exact_match_rule()


if __name__ == '__main__':
    main()