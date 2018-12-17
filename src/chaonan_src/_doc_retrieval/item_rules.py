#!/usr/bin/env python
"""Item rule builder
"""
from copy import copy, deepcopy
import json

import numpy as np
from flashtext import KeywordProcessor
import inflection

import config
from drqa_yixin.tokenizers import CoreNLPTokenizer, set_default
from utils import common, wiki_term_builder
from utils.text_clean import STOPWORDS
from doc_retrieval.fast_key_word_matching_v1_3 import \
     id_dict_key_word_expand, \
     set_priority, \
     load_data, \
     check_inside_paretheses_overlap, \
     load_keyword_dict_v1_3
from doc_retrieval.fast_key_word_matching_v1_3 import \
     get_words_inside_parenthese as extract_par
from doc_retrieval.fast_key_word_matching_v1_3 import \
     build_flashtext_processor_with_prioritized_kw_dict as build_processor
from utils import fever_db
from utils.fever_db import convert_brc
from chaonan_src._utils.doc_utils import \
     get_default_tfidf_ranker_args, \
     reverse_convert_brc, \
     DocIDTokenizer
from chaonan_src._doc_retrieval.google_querier import GoogleQuerier
from chaonan_src._utils.wiki_pageview_utils import WikiPageviews
from chaonan_src import _config


__all__ = ['KeywordRuleBuilder',
           'DocIDRuleBuilder',
           'SentenceSimilarityRuleBuilder',
           'ItemRuleBuilderBase',
           'ItemRuleBuilder',
           'ItemRuleBuilderLower',
           'ItemRuleBuilderRawID',
           ]
__author__ = ['chaonan99', 'yixin1']


class KeywordRuleBuilder(object):
    """KeywordRuleBuilder applies post processing rules on keyword processor
    """
    @classmethod
    def __essential_remove(cls, keyword_processor, remove_str):
        remove_set = copy(keyword_processor[remove_str])
        if remove_set is not None:
            result_set = set([it for it in remove_set if it[0] != remove_str])
            if len(result_set) == 0:
                keyword_processor.remove_keyword(remove_str)
            else:
                keyword_processor[remove_str] = result_set

    @classmethod
    def eliminate_pure_digits_in_place(cls, keyword_processor):
        for i in range(100000):
            cls.__essential_remove(keyword_processor, str(i))

    @classmethod
    def eliminate_ordinals_in_place(cls, keyword_processor):
        for i in range(1000):
            cls.__essential_remove(keyword_processor, inflection.ordinalize(i))

    @classmethod
    def eliminate_stop_words_in_place(cls, keyword_processor):
        for w in STOPWORDS:
            cls.__essential_remove(keyword_processor, w)
            cls.__essential_remove(keyword_processor, w.title())



class DocIDRuleBuilder(object):
    """DocIDRuleBuilder contains docid based rules, including parentheses
    overlapping, enhancement for film, TV series, ... and others
    """
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
        addup_score = 0.2 if 'film'  in extract_par(self.doc_id_tokens) \
                 else 0.1 if 'album' in extract_par(self.doc_id_tokens) \
                          or 'TV'    in extract_par(self.doc_id_tokens) \
                 else 0
        self.priority += addup_score
        return self

    @property
    def id_prio_tuple(self):
        return (self.doc_id, self.priority)


class SentenceSimilarityRuleBuilder(object):
    """SentenceSimilarityRuleBuilder is used for ad hoc enhancement
    """
    def __init__(self):
        self.tfidf_args = get_default_tfidf_ranker_args()

    def tfidf_similarity(self, sent_list, item, k=5):
        ranker = OnlineTfidfDocRanker(self.tfidf_args,
                                      self.tfidf_args.hash_size,
                                      self.tfidf_args.ngram,
                                      sent_list)
        indexes, scores = ranker.closest_docs(item['claim'], k=5)
        return np.array(indexes), scores

    def first_sent_similarity(self, sent_list, item, *args):
        return self.preceding_sent_similarity(sent_list, item, 1)

    def preceding_sent_similarity(self, sent_list, item, k=5, *args):
        return self.preceding_sent_weighed_similarity(sent_list,
                                                      item,
                                                      k=5,
                                                      start=1,
                                                      end=1)

    def preceding_sent_weighed_similarity(self, sent_list,
                                                item,
                                                k=5,
                                                start=1.1,
                                                end=0.8,
                                                *args):
        k = min(len(sent_list), k)
        return np.arange(k), np.geomspace(start, end, k)


class ItemRuleBuilderBase:
    """ItemRuleBuilderBase is the base class for item rule builder
    """
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

    def _build_kp(self, case_sensitive=True):
        ## Prepare tokenizer and flashtext keyword processor
        keyword_processor = KeywordProcessor(case_sensitive=case_sensitive)
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
        KeywordRuleBuilder.eliminate_stop_words_in_place(keyword_processor)

        return keyword_processor

    def build_kp(self, case_sensitive=True):
        return self._build_kp(case_sensitive)

    def _keyword_match(self, claim, raw_set=False, custom_kp=None):
        kp = self.keyword_processor if custom_kp is None else custom_kp
        finded_keys = kp.extract_keywords(claim)
        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys = set.union(*finded_keys)
        return finded_keys

    def get_token_lemma_from_claim(self, claim):
        claim_norm = wiki_term_builder.normalize(claim)
        claim_tok_r = self.tokenizer.tokenize(claim_norm)
        claim_tokens = claim_tok_r.words()
        claim_lemmas = claim_tok_r.lemmas()
        return claim_tokens, claim_lemmas

    @classmethod
    def get_all_docid_in_evidence(cls, evidence):
        return [iii for i in evidence \
                    for ii in i \
                    for iii in ii \
                    if type(iii) == str]

    @property
    def rules(self):
        return lambda x: x


class ItemRuleBuilder(ItemRuleBuilderBase):
    """ItemRuleBuilder contains basic document retrieval rules
    """
    def __init__(self, tokenizer=None, keyword_processor=None):
        super().__init__(tokenizer, keyword_processor)
        self.sent_sim = SentenceSimilarityRuleBuilder()
        self.docid_tokenizer = DocIDTokenizer(case_insensitive=True)
        self.google_querier = GoogleQuerier(self.keyword_processor)

    def exact_match_rule(self, item):
        claim_tokens, claim_lemmas = \
            self.get_token_lemma_from_claim(item['claim'])
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
            docid_rule_builder.tokenize_docid(id_prio_tuple,
                                              self.docid_tokenizer)\
                              .parentheses_overlap_rule()\
                              .common_word_rule()
            item['prioritized_docids'][i] = docid_rule_builder.id_prio_tuple
        return self

    def eliminate_the_rule(self):
        return self.eliminate_start_words_rule(starts=['The'])

    def eliminate_articles_rule(self):
        return self.eliminate_start_words_rule(starts=['The', 'A', 'An'])

    def eliminate_start_words_rule(self, starts=['The'], modify_pdocid=False):
        item = self.item
        claim_tokens = copy(item['claim_tokens'])
        finded_keys  = item['prioritized_docids']
        if claim_tokens[0] in starts:
            claim_tokens[1] = claim_tokens[1].title()
            claim = ' '.join(claim_tokens[1:])
            fk_new = self._keyword_match(claim)
            finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(finded_keys)
        return self

    def singularize_rule(self):
        """Singularize words
        """
        item = self.item
        if len(item['prioritized_docids']) < 1:
            claim_tokens = item['claim_tokens']
            # finded_keys  = item['prioritized_docids']
            claim_tokens = [inflection.singularize(c) for c in claim_tokens]
            claim = ' '.join(claim_tokens)
            fk_new = self._keyword_match(claim)
            # finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(fk_new)
        return self

    def google_query_rule(self):
        item = self.item
        docid_dict = {k: v for k, v in item['prioritized_docids']}
        esn = sum([v > 1 for v in docid_dict.values()])
        if (len(item['prioritized_docids']) > 15 \
          and esn < 5) \
          or len(item['prioritized_docids']) < 1:
            self.google_querier.get_google_docid(item)
            for k in item['google_docids']:
                docid_dict[k] = 6.0
            item['prioritized_docids'] = [(k, v) for k, v in docid_dict.items()]
        return self

    def expand_from_doc_rule(self):
        """Current method: if 'prioritized_docids' is shorter than 5, then
        expand every found ID by extract the document, find some highly-scored
        (currently tf-idf score) sentences, find links in them and append those
        documents.

        Discussions on some variations
        ------------------------------
        1. Can use other types of sentence similarity score
        2. Can (kind of) combine sentence score into priority
        3. Match appears first can have higher score propagated

        """
        if not hasattr(self, 'cursor'):
            self.cursor = fever_db.get_cursor()
        item = self.item

        if len(item['prioritized_docids']) < 2:
            # print(f"Query tf-idf... because length={len(item['prioritized_docids'])}")
            new_pdocids = copy(item['prioritized_docids'])
            for docid, priority in item['prioritized_docids']:
                # print(f"Query tf-idf for {docid}")
                sent_list, id_list, sent_links = \
                    fever_db.get_all_sent_by_doc_id(self.cursor,
                                                    docid,
                                                    with_h_links=True)
                # indexes, scores = \
                #     self.sent_sim.preceding_sent_similarity(sent_list,
                #                                         item['claim'])
                indexes, scores = \
                    self.sent_sim.tfidf_similarity(sent_list,
                                                        item['claim'])

                high_tfidf_indexes = indexes[scores > 3.0]
                if len(high_tfidf_indexes) > 0:
                    all_links = np.array(sent_links)[high_tfidf_indexes]
                    all_links = [ii for i in all_links for ii in i]  # flatten links
                    all_links = np.array(all_links)
                    all_links = all_links.reshape(-1, 2)[:, 1]
                    all_links = list(map(reverse_convert_brc, all_links))
                    new_pdocids.extend([(id_link, 1.0*priority) \
                        for id_link in all_links])
            item['prioritized_docids'] = new_pdocids
        return self

    def expand_from_preext_sent_rule(self):
        print("Enter")
        if not hasattr(self, 'cursor'):
            self.cursor = fever_db.get_cursor()
        if not hasattr(self, 'preext_sent_dict'):
            d_list = load_data(config.RESULT_PATH / \
                "sent_retri_nn/2018_07_17_16-34-19_r/dev_scale(0.1).jsonl")
            self.preext_sent_dict = {item['id']: item for item in d_list}
        item = self.item

        if len(item['prioritized_docids']) < 5:
            new_pdocids = copy(item['prioritized_docids'])
            sent_ids = self.preext_sent_dict[item['id']]['predicted_sentids']
            for sent_id in sent_ids:
                docid, sent_ind = sent_id.split('<SENT_LINE>')
                sent_ind = int(sent_ind)
                id_list, sent_list, sent_links = \
                    fever_db.get_evidence(self.cursor,
                                          docid,
                                          sent_ind)
                sent_links = json.loads(sent_links)
                all_links = np.array(sent_links)
                all_links = np.array(all_links)
                all_links = all_links.reshape(-1, 2)[:, 1]
                all_links = list(map(reverse_convert_brc, all_links))
                new_pdocids.extend([(id_link, 1.0) \
                    for id_link in all_links])
            item['prioritized_docids'] = new_pdocids

    @property
    def rules(self):
        return lambda x: self.exact_match_rule(x)\
                             .eliminate_articles_rule()\
                             .docid_based_rule()\
                             .singularize_rule()\
                             # .google_query_rule()\
                             # .expand_from_preext_sent_rule()\
                             #


class ItemRuleBuilderLower(ItemRuleBuilder):
    """docstring for ItemRuleBuilderLower"""
    def __init__(self, tokenizer=None, keyword_processor=None, kp_lower=None):
        super(ItemRuleBuilderLower, self).__init__(tokenizer, keyword_processor)
        self.kp_lower = self.build_kp(case_sensitive=False) \
                        if kp_lower is None else kp_lower
        # self.freq_words = [l.rstrip() for l in open(_config.freq_words_path)]

    def _keyword_match_lower(self, claim):
        return self._keyword_match(claim, custom_kp=self.kp_lower)

    def lower_match_rule(self):
        item = self.item
        if len(item['prioritized_docids']) < 1:
            cts = [c for c in item['claim_tokens'] if c not in STOPWORDS]
            # cts = [c for c in item['claim_tokens'] if c not in self.freq_words]
            claim = ' '.join(cts)
            finded_keys = self._keyword_match_lower(claim)
            item['prioritized_docids'].extend(list(finded_keys))
        return self

    def google_disambiguious_rule(self):
        item = self.item
        return self

    def more_rules(self):
        return self

    @property
    def rules(self):
        return lambda x: self.exact_match_rule(x)\
                             .eliminate_articles_rule()\
                             .docid_based_rule()\
                             .singularize_rule()\
                             .lower_match_rule()\
                             # .more_rules()


class ItemRuleBuilderRawID(ItemRuleBuilder):
    """docstring for ItemRuleBuilderRawID"""
    def __init__(self, tokenizer=None, keyword_processor=None):
        super(ItemRuleBuilderRawID, self).__init__(tokenizer,
                                                   keyword_processor)

    def build_kp(self, case_sensitive=False):
        return self._build_kp(case_sensitive)

    def _recursive_key_matcher(self, claim, fkd=None):
        fkd = {} if fkd is None else fkd
        finded_keys_dict = self.google_querier.get_keywords(claim)

        for key, value in finded_keys_dict.items():
            if key.lower() in STOPWORDS:
                continue

            ## First letter is case sensitive
            value = [v for v in value if v[0][0] == key[0]]

            if len(value) == 0:
                ## This seems to be a bug, for example, for claim
                ## "The hero of the Odyssey is Harry Potter.", it will first
                ## match "the Odyssey" and failed to match first letter
                ## "The_Odyssey" and "the Odyssey" will be ignored.
                ## But 2 actually performs best on dev so we just keep it here
                key_tokens = key.split(' ')
                if len(key_tokens) > 2:
                    key_tokens[1] = key_tokens[1].title()
                    self._recursive_key_matcher(' '.join(key_tokens[1:]), fkd)
            else:
                fkd.update({key:value})

        return fkd

    def _keyword_match(self, claim, raw_set=False, custom_kp=None):
        kp = self.keyword_processor if custom_kp is None else custom_kp
        if not raw_set:
            finded_keys = kp.extract_keywords(claim)
            if isinstance(finded_keys, list) and len(finded_keys) != 0:
                finded_keys = set.union(*finded_keys)
            return finded_keys
        else:
            finded_keys_dict = self.google_querier.get_keywords(claim)
            finded_keys_dict = self._recursive_key_matcher(claim)
            finded_keys = finded_keys_dict.values()
            finded_keys = set([i for ii in finded_keys for i in ii]) \
                          if len(finded_keys) > 0 else set(finded_keys)

            return finded_keys_dict, finded_keys

    def google_query_rule(self):
        item = self.item
        if len(item['prioritized_docids']) > 40:
            # all_keys = item['structured_docids'].keys()
            item['google_docids'] = []
            matched_docid = self.google_querier\
                .google_it(item['processed_claim'])
            # item['prioritized_docids'].append((matched_docid, 6.0))
            # Consume redundent keywords
            print(item['processed_claim'])
            print(matched_docid)
            if matched_docid is not None:
                fkd_new = {}
                key_remains = []
                for key, value in item['structured_docids'].items():
                    key_tokens = key.split(' ')
                    if not np.all(list(map(lambda x: x in matched_docid,
                                           key_tokens))):
                        key_remains.append(key)
                        fkd_new.update({key: value})
                item['structured_docids'] = fkd_new
                finded_keys = fkd_new.values()
                finded_keys = set([i for ii in finded_keys for i in ii]) \
                              if len(finded_keys) > 0 else set(finded_keys)
                item['prioritized_docids'] = list(finded_keys)
                item['prioritized_docids'].append((matched_docid, 6.0))
        return self

    def test_recursive_match(self, claim):
        return self._recursive_key_matcher(claim)

    def exact_match_rule(self, item):
        claim_tokens, claim_lemmas = \
            self.get_token_lemma_from_claim(item['claim'])
        claim = ' '.join(claim_tokens)

        finded_keys_dict, finded_keys = self._keyword_match(claim, raw_set=True)

        item['prioritized_docids'] = list(finded_keys)
        item['structured_docids'] = finded_keys_dict
        # print(finded_keys_dict)
        item['claim_lemmas'] = claim_lemmas
        item['claim_tokens'] = claim_tokens
        item['processed_claim'] = claim
        self.item = item
        return self

    def pageview_rule(self):
        """Assign high priority to frequently viewed pages
        """
        if not hasattr(self, 'wiki_pv'):
            print("Reload wiki pageview dict")
            self.wiki_pv = WikiPageviews()

        item = self.item
        docid_groups = [[i[0] for i in it] \
                        for _, it in item['structured_docids'].items()]
        changed = False
        for key, group_prio_docids in item['structured_docids'].items():
            group_docids = [it[0] for it in group_prio_docids]
            if len(group_docids) > 1:
                changed = True
                all_scores = map(lambda x: self.wiki_pv[convert_brc(x)],
                                 group_docids)
                all_scores = np.array(list(all_scores))
                prios = np.argsort(all_scores)[::-1]
                new_gpd = []
                for i, p in enumerate(prios):
                    # new_gpd.append((group_prio_docids[p][0],
                    #                 group_prio_docids[p][1] + \
                    #                     max(1.0 - i*0.2, 0)))
                    new_gpd.append((group_prio_docids[p][0],
                                    max(1.0 - i*0.2, 0)))
                item['structured_docids'][key] = new_gpd

        if changed:
            finded_keys = item['structured_docids'].values()
            finded_keys = set([i for ii in finded_keys for i in ii]) \
                          if len(finded_keys) > 0 else set(finded_keys)
            item['prioritized_docids'] = list(finded_keys)
        return self

    def eliminate_start_words_rule(self, starts=['The']):
        item = self.item
        claim_tokens = copy(item['claim_tokens'])
        finded_keys  = item['prioritized_docids']
        if claim_tokens[0] in starts:
            claim_tokens[1] = claim_tokens[1].title()
            claim = ' '.join(claim_tokens[1:])
            fkd_new, fk_new = self._keyword_match(claim, raw_set=True)
            finded_keys = set(finded_keys) | set(fk_new)
            item['structured_docids'].update(fkd_new)
            item['prioritized_docids'] = list(finded_keys)
        return self

    def singularize_rule(self):
        item = self.item
        if len(item['prioritized_docids']) < 1:
            claim_tokens = item['claim_tokens']
            # finded_keys  = item['prioritized_docids']
            claim_tokens = [inflection.singularize(c) for c in claim_tokens]
            claim = ' '.join(claim_tokens)
            fkd_new, fk_new = self._keyword_match(claim, raw_set=True)
            # finded_keys = set(finded_keys) | set(fk_new)
            item['prioritized_docids'] = list(fk_new)
            item['structured_docids'] = fkd_new
        return self

    def expand_from_preext_sent_rule(self):
        if not hasattr(self, 'cursor'):
            self.cursor = fever_db.get_cursor()
        if not hasattr(self, 'preext_sent_dict'):
            d_list = load_data(config.RESULT_PATH / \
                "sent_retri_nn/2018_07_17_16-34-19_r/dev_scale(0.1).jsonl")
            self.preext_sent_dict = {item['id']: item for item in d_list}
        item = self.item

        # if len(item['prioritized_docids']) < 5:
        new_pdocids = copy(item['prioritized_docids'])
        sent_ids = self.preext_sent_dict[item['id']]['predicted_sentids']
        for sent_id in sent_ids:
            docid, sent_ind = sent_id.split('<SENT_LINE>')
            sent_ind = int(sent_ind)
            id_list, sent_list, sent_links = \
                fever_db.get_evidence(self.cursor,
                                      docid,
                                      sent_ind)
            sent_links = json.loads(sent_links)
            all_links = np.array(sent_links)
            all_links = np.array(all_links)
            all_links = all_links.reshape(-1, 2)[:, 1]
            all_links = list(map(reverse_convert_brc, all_links))
            new_pdocids.extend([(id_link, 1.0) \
                for id_link in all_links])
        item['prioritized_docids'] = new_pdocids
        return self

    @property
    def rules(self):
        return lambda x: self.exact_match_rule(x)\
                             .docid_based_rule()\
                             .singularize_rule()\
                             .eliminate_articles_rule()\
                             .pageview_rule()\
                             # .google_query_rule()\
                             # .more_rules()
                             # .expand_from_preext_sent_rule()\


class ItemRuleBuilderRawIDMultiMatch(ItemRuleBuilderRawID):
    """docstring for ItemRuleBuilderRawIDMultiMatch"""
    def __init__(self, tokenizer=None, keyword_processor=None):
        super(ItemRuleBuilderRawIDMultiMatch, self).__init__(tokenizer,
                                                             keyword_processor)

    def _recursive_key_matcher(self, claim):
        keyword_list = self.keyword_processor.extract_keywords(claim,
                                                               span_info=True)
        finded_keys_dict = {claim[it[1]:it[2]]:it[0] for it in keyword_list}

        fkd = {}

        for value, start, end in keyword_list:
            key = claim[start:end]
            if key.lower() in STOPWORDS:
                continue

            ## First letter is case sensitive
            new_value = [v for v in value if v[0][0] == key[0]]

            if len(new_value) == 0:
                # print(value)
                ## Failed
                key_tokens = key.split(' ')
                if len(key_tokens) > 2:
                    new_claim = ' '.join([claim[:start-1],
                                          *key_tokens[1:],
                                          claim[end+1:]])
                    return self._recursive_key_matcher(new_claim)
            else:
                fkd.update({key:new_value})

        return fkd

    # def _keyword_match(self, claim, raw_set=False, custom_kp=None):
    #     kp = self.keyword_processor if custom_kp is None else custom_kp
    #     if not raw_set:
    #         finded_keys = kp.extract_keywords(claim)
    #         if isinstance(finded_keys, list) and len(finded_keys) != 0:
    #             finded_keys = set.union(*finded_keys)
    #         return finded_keys
    #     else:
    #         finded_keys_dict = self.google_querier.get_keywords(claim)
    #         finded_keys_dict = self._recursive_key_matcher(claim, {})
    #         finded_keys = finded_keys_dict.values()
    #         finded_keys = set([i for ii in finded_keys for i in ii]) \
    #                       if len(finded_keys) > 0 else set(finded_keys)

    #         return finded_keys_dict, finded_keys

    @classmethod
    def _get_pdocid_from_structure(cls, finded_keys_dict):
        finded_keys = finded_keys_dict.values()
        return list(set([i for ii in finded_keys for i in ii]) \
                           if len(finded_keys) > 0 else set(finded_keys))

    def eliminate_articles_rule(self):
        return self.eliminate_start_words_rule(starts=['The', 'A', 'An', 'One'])

    def eliminate_start_words_rule(self, starts=['The'], modify_pdocid=False):
        item = self.item
        claim_tokens = copy(item['claim_tokens'])
        # finded_keys  = item['prioritized_docids']
        if claim_tokens[0] in starts:
            item['structured_docids'].pop(claim_tokens[0], None)
            claim_tokens[1] = claim_tokens[1].title()
            claim = ' '.join(claim_tokens[1:])
            fkd_new, fk_new = self._keyword_match(claim, raw_set=True)
            item['structured_docids'].update(fkd_new)
            item['prioritized_docids'] = \
                self._get_pdocid_from_structure(item['structured_docids'])

        return self
