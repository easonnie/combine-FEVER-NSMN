#!/usr/bin/env python
"""Document retrieval experiments for `FEVER <http://fever.ai/>`_


      ┌─┐       ┌─┐ + +
   ┌──┘ ┴───────┘ ┴──┐++
   │                 │
   │       ───       │++ + + +
   ███████───███████ │+
   │                 │+
   │       ─┴─       │
   │                 │
   └───┐         ┌───┘
       │         │
       │         │   + +
       │         │
       │         └──────────────┐
       │                        │
       │                        ├─┐
       │                        ┌─┘
       │                        │
       └─┐  ┐  ┌───────┬──┐  ┌──┘  + + + +
         │ ─┤ ─┤       │ ─┤ ─┤
         └──┴──┘       └──┴──┘  + + + +
             God bless
               No BUG

"""

from time import time
import io
import sys
from copy import copy

import numpy as np
from flashtext import KeywordProcessor
import inflection

import config
from sentence_retrieval.sent_tfidf import OnlineTfidfDocRanker
from drqa_yixin.tokenizers import CoreNLPTokenizer, set_default
import utils
from utils import common
from utils.c_scorer import fever_score, check_doc_id_correct
from doc_retrieval.fast_key_word_matching_v1_3 import \
     id_dict_key_word_expand, \
     set_priority, \
     load_data, \
     get_words_inside_parenthese, \
     check_inside_paretheses_overlap, \
     load_keyword_dict_v1_3
from doc_retrieval.fast_key_word_matching_v1_3 import \
     build_flashtext_processor_with_prioritized_kw_dict as build_processor
from utils import fever_db
from chaonan_src._utils.doc_utils import \
     reverse_convert_brc, \
     get_default_tfidf_ranker_args, \
     DocIDTokenizer


__all__ = ['KeywordRuleBuilder',
           'DocIDRuleBuilder',
           'ItemRuleBuilderBase',
           'ItemRuleBuilder',
           'DocRetrievalExperiment']
__author__ = ['chaonan99', 'yixin1']


class KeywordRuleBuilder(object):
    """KeywordRuleBuilder applies post processing rules on keyword processor"""

    @classmethod
    def eliminate_pure_digits_in_place(cls, keyword_processor):
        for i in range(100000):
            numstr = str(i)
            numset = copy(keyword_processor[numstr])
            if numset is not None:
                result_set = set([it for it in numset if it[0] != numstr])
                if len(result_set) == 0:
                    keyword_processor.remove_keyword(numstr)
                else:
                    keyword_processor[numstr] = result_set

    @classmethod
    def eliminate_ordinals_in_place(cls, keyword_processor):
        for i in range(1000):
            numstr = inflection.ordinalize(i)
            numset = copy(keyword_processor[numstr])
            if numset is not None:
                result_set = set([it for it in numset if it[0] != numstr])
                if len(result_set) == 0:
                    keyword_processor.remove_keyword(numstr)
                else:
                    keyword_processor[numstr] = result_set


class DocIDRuleBuilder(object):
    """docstring for DocIDRuleBuilder"""
    def __init__(self, claim_tokens, claim_lemmas):
        self.claim_tokens = claim_tokens
        self.claim_lemmas = claim_lemmas

    def tokenize_docid(self, id_prio_tuple, docid_tokenizer):
        self.doc_id, self.priority = id_prio_tuple
        self.doc_id_tokens, self.doc_id_lemmas = \
            docid_tokenizer.tokenize_docid(self.doc_id.lower())
        return self

    def parentheses_overlap_rule(self):
        if self.priority == 1.0:
            self.priority = check_inside_paretheses_overlap(self.doc_id_tokens,
                                                            self.doc_id_lemmas,
                                                            self.claim_tokens,
                                                            self.claim_lemmas)
        return self

    def common_word_rule(self):
        addup_score = 0.2 if 'film'      in get_words_inside_parenthese(self.doc_id_tokens) \
                 else 0.1 if 'album'     in get_words_inside_parenthese(self.doc_id_tokens) \
                          or 'TV_series' in get_words_inside_parenthese(self.doc_id_tokens) \
                 else 0
        self.priority += addup_score
        return self

    @property
    def id_prio_tuple(self):
        return (self.doc_id, self.priority)


class ItemRuleBuilderBase:
    """docstring for ItemRuleBuilderBase"""
    def __init__(self, tokenizer=None, keyword_processor=None):
        self.tokenizer = self.initialize_tokenizer() if tokenizer is None \
                                                     else tokenizer
        self.keyword_processor = self.build_kp() if keyword_processor is None \
                                                 else keyword_processor

    def initialize_tokenizer(self):
        snlp_path = str(config.PRO_ROOT / \
            'dep_packages/stanford-corenlp-full-2017-06-09/*')
        set_default('corenlp_classpath', snlp_path)
        return CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    def build_kp(self):
        ## Prepare tokenizer and flashtext keyword processor
        keyword_processor = KeywordProcessor(case_sensitive=True)
        id_to_key_dict = load_keyword_dict_v1_3(
            config.DATA_ROOT / "id_dict.jsonl", filtering=True)
        exact_match_rule_dict = set_priority(id_to_key_dict, priority=5.0)
        noisy_key_dict = id_dict_key_word_expand(id_to_key_dict,
                                                 create_new_key_word_dict=True)
        noisy_parenthese_rule_dict = set_priority(noisy_key_dict, priority=1.0)

        build_processor(keyword_processor,
                        exact_match_rule_dict)
        build_processor(keyword_processor,
                        noisy_parenthese_rule_dict)

        ## Change priorities of digital numbers
        KeywordRuleBuilder.eliminate_pure_digits_in_place(keyword_processor)
        KeywordRuleBuilder.eliminate_ordinals_in_place(keyword_processor)

        return keyword_processor

    def _keyword_match(self, claim):
        finded_keys = self.keyword_processor.extract_keywords(claim)
        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys = set.union(*finded_keys)
        return finded_keys

    def get_token_lemma_from_claim(self, claim):
        claim_norm = utils.wiki_term_builder.normalize(claim)
        claim_tok_r = self.tokenizer.tokenize(claim_norm)
        claim_tokens = claim_tok_r.words()
        claim_lemmas = claim_tok_r.lemmas()
        return claim_tokens, claim_lemmas

    @classmethod
    def get_all_docid_in_evidence(cls, evidence):
        return [iii for i in evidence for ii in i for iii in ii if type(iii) == str]

    @property
    def rules(self):
        return lambda x: x


class ItemRuleBuilder(ItemRuleBuilderBase):
    """docstring for ItemRuleBuilder"""
    def __init__(self, tokenizer=None, keyword_processor=None):
        super().__init__(tokenizer, keyword_processor)
        self.tfidf_args = get_default_tfidf_ranker_args()
        self.docid_tokenizer = DocIDTokenizer(case_insensitive=True)

    def exact_match_rule(self, item):
        claim_tokens, claim_lemmas = self.get_token_lemma_from_claim(item['claim'])
        claim = ' '.join(claim_tokens)

        finded_keys = self._keyword_match(claim)

        item['prioritized_docids'] = list(finded_keys)
        item['claim_lemmas'] = claim_lemmas
        item['claim_tokens'] = claim_tokens
        item['processed_claim'] = claim
        self.item = item
        return self

    def docid_based_rule(self):
        item = self.item
        assert 'prioritized_docids' in item, 'Apply exact match rule first!'
        for i, id_prio_tuple in enumerate(item['prioritized_docids']):
            docid_rule_builder = DocIDRuleBuilder(item['claim_tokens'],
                                                  item['claim_lemmas'])
            docid_rule_builder.tokenize_docid(id_prio_tuple, self.docid_tokenizer)\
                              .parentheses_overlap_rule()\
                              .common_word_rule()
            item['prioritized_docids'][i] = docid_rule_builder.id_prio_tuple
        return self

    def eliminate_the_rule(self):
        """When the first token is 'The', it is sometimes not part of a doc id
        This helps from 87.78 to 88.79
        """
        item = self.item
        claim_tokens = item['claim_tokens']
        finded_keys  = item['prioritized_docids']
        if claim_tokens[0] == 'The':
            claim_tokens[1] = claim_tokens[1].title()
            claim = ' '.join(claim_tokens[1:])
            fk_new = self._keyword_match(claim)
            finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(finded_keys)
        return self

    @property
    def rules(self):
        return lambda x: self.exact_match_rule(x)\
                             .eliminate_the_rule()\
                             .docid_based_rule()


class DocRetrievalExperiment(object):
    """docstring for DocRetrievalExperiment"""
    def __init__(self, item_rb=None):
        self.item_rb = ItemRuleBuilder() if item_rb is None else item_rb

    def sample_answer_with_priority(self, d_list, top_k=5):

        for i, item in enumerate(d_list):
            if (i+1) % 50 == 0 or (i+1) == len(d_list):
                print(f"\rProcessed {i+1}/{len(d_list)}",end='',flush=True)
            self.item_rb.rules(item)
            item['predicted_docids'] = \
                list(set([k for k, v in sorted(item['prioritized_docids'],
                                               key=lambda x: (-x[1], x[0]))][:top_k]))

    @classmethod
    def eval(self, d_list):
        eval_mode = {'check_doc_id_correct': True, 'standard': False}
        return fever_score(d_list, d_list, mode=eval_mode, verbose=False)

    @classmethod
    def extract_failure(cls, d_list):
        correct = [check_doc_id_correct(item) for item in d_list]
        return np.array(d_list)[~np.array(correct)].tolist()

    @classmethod
    def sample_item(cls, d_list):
        return np.random.choice(d_list)


def main():
    doc_exp = DocRetrievalExperiment()
    # d_list = load_data(config.FEVER_DEV_JSONL)
    d_list = load_data(config.FEVER_TRAIN_JSONL)

    start = time()
    doc_exp.sample_answer_with_priority(d_list)
    end = time()
    print(f"Time usage: {end-start} s")

    result = doc_exp.eval(d_list)
    print(result)

    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()