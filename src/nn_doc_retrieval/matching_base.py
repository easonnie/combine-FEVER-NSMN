from chaonan_src._doc_retrieval.item_rules import KeywordRuleBuilder
from utils import common
from utils import c_scorer
from flashtext import KeywordProcessor
import config
from doc_retrieval.fast_key_word_matching_v1_3 import \
     build_flashtext_processor_with_prioritized_kw_dict as build_processor

from doc_retrieval.fast_key_word_matching_v1_3 import \
     id_dict_key_word_expand, \
     set_priority, \
     load_data, \
     check_inside_paretheses_overlap, \
     load_keyword_dict_v1_3

if __name__ == '__main__':
    case_sensitive = True
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