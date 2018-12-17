from collections import Counter

from flashtext import KeywordProcessor
import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
import utils.wiki_term_builder
from tqdm import tqdm
import json
from utils.fever_db import get_all_doc_ids, convert_brc
from utils.c_scorer import fever_score
from utils import text_clean
from pathlib import Path
import copy
import utils


def build_keyword_dict(did_list, tokenizer, out_filename):
    out_f = open(out_filename, encoding='utf-8', mode='w') if out_filename is not None else None
    for doc_id in tqdm(did_list):
        item = dict()
        item['docid'] = doc_id
        item['keys'], item['lemmas'], item['entities'] = did_to_keys(doc_id, tokenizer=tokenizer)

        if out_f is not None:
            out_f.write(json.dumps(item) + '\n')

    out_f.flush()
    out_f.close()


def load_keyword_dict(in_filename, filtering=False):
    id_to_key_dict = dict()
    with open(in_filename, encoding='utf-8', mode='r') as in_f:
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            if filtering and text_clean.filter_document_id(item['docid']):
                continue
            id_to_key_dict[item['docid']] = item['keys']

    return id_to_key_dict


def set_priority(id_to_key_dict, priority):
    """
    {'document_id': [key_trigger_words]}
    -> {'document_id': [(key_trigger_words, priority)]}
    :param priority: The priority
    :param id_to_key_dict:
    :return:
    """
    prioritied_dict = dict()
    for doc_id, keys in id_to_key_dict.items():
        prioritied_dict[doc_id] = [(key, priority) for key in keys]

    return prioritied_dict


def build_flashtext_processor_with_prioritized_kw_dict(keyword_processor, keyword_dict):
    """
    This method convert keyword dictionary to a flashtext consumeable format and build the keyword_processor
    :param keyword_processor:
    :param keyword_dict:
    { 'document_id' : [(key_trigger_words, priority)]

    flashtext consumeable
    { 'key_trigger_word', set((document_id, priority))}
    :return:
    """
    for doc_id, kwords in tqdm(keyword_dict.items()):
        for kw, priority in kwords:
            if kw in keyword_processor:
                # If doc_id exist:
                found = False

                for exist_doc_id, exist_priority in keyword_processor[kw]:
                    if exist_doc_id == doc_id:
                        # Update the priority by remove the old and add the new
                        keyword_processor[kw].remove((exist_doc_id, exist_priority)) # Remove original
                        keyword_processor[kw].add(doc_id, max(exist_priority, priority)) # Add new
                        found = True
                        break

                # If doc_id not found
                if not found:
                    keyword_processor[kw].add((doc_id, priority))

            else:
                keyword_processor.add_keyword(kw, {(doc_id, priority)})


def did_to_keys(doc_id, tokenizer=None):
    doc_id = convert_brc(doc_id)
    doc_id = doc_id.replace('_', ' ')
    id_keys = []
    # id_keys.append(doc_id)
    lemmas = None
    entities = None
    if tokenizer is not None:
        tok_r = tokenizer.tokenize(doc_id)
        to_key = ' '.join(tok_r.words())
        id_keys.append(to_key)

        lemmas = tok_r.lemmas()
        entities = tok_r.entity_groups()

    return list(set(id_keys)), lemmas, entities


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def save_intermidiate_results(d_list, out_filename):
    out_filename.parent.mkdir(exist_ok=False)
    with open(out_filename, encoding='utf-8', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')


def sample_answer(d_list, tokenizer, keyword_p):
    count = 0
    num_of_select = Counter()
    print("Build results file...")
    for item in tqdm(d_list):
        # evidences = list(check_and_clean_evidence(item))
        # print(evidences[0].evidences_list)
        # item['predicted_label'] = item['label']
        # item['predicted_evidence'] = list(map(list, evidences[0].evidences_list))
        # p_e = list(map(list, evidences[0].evidences_list))
        # item['predicted_evidence'] = p_e
        tokens = tokenizer.tokenize(item['claim']).words()
        claim = ' '.join(tokens)
        # print(claim)
        finded_keys = keyword_p.extract_keywords(claim)

        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys = set.union(*finded_keys)

        item['predicted_docids'] = list(finded_keys)

        num_of_select.update([len(finded_keys)])

    print(num_of_select)
    print(num_of_select.most_common())
    print(count)


def sample_answer_with_priority(d_list, tokenizer, keyword_p, top_k=5):
    count = 0
    num_of_select = Counter()
    print("Build results file...")
    for item in tqdm(d_list):
        # evidences = list(check_and_clean_evidence(item))
        # print(evidences[0].evidences_list)
        # item['predicted_label'] = item['label']
        # item['predicted_evidence'] = list(map(list, evidences[0].evidences_list))
        # p_e = list(map(list, evidences[0].evidences_list))
        # item['predicted_evidence'] = p_e
        tokens = tokenizer.tokenize(utils.wiki_term_builder.normalize(item['claim'])).words()
        claim = ' '.join(tokens)
        # print(claim)
        item['prioritized_docids'] = []

        finded_keys = keyword_p.extract_keywords(claim)

        if isinstance(finded_keys, list) and len(finded_keys) != 0:
            finded_keys = set.union(*finded_keys)

        # TOD We can add something here to fine-select the document
        item['prioritized_docids'] = list(finded_keys)

        item['predicted_docids'] = \
            list(set([k for k, v in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0]))][:top_k]))

        num_of_select.update([len(item['predicted_docids'])])

    print(num_of_select)
    print(num_of_select.most_common())
    print(count)


def used_func_for_building_normalized_key_word_index_for_docids():
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    did_list = get_all_doc_ids(str(config.FEVER_DB), max_ind=None)

    build_keyword_dict(did_list, tok,
                       config.DATA_ROOT / "id_dict.jsonl")


def used_func_for_fast_key_word_matching():
    # Load tokenizer
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    keyword_processor = KeywordProcessor(case_sensitive=True)
    id_to_key_dict = load_keyword_dict(config.DATA_ROOT / "id_dict.jsonl")

    # Write this in a for loop to keep track of the progress
    for clean_name, keywords in tqdm(id_to_key_dict.items()):
        if not isinstance(keywords, list):
            raise AttributeError("Value of key {} should be a list".format(clean_name))

        for keyword in keywords:
            keyword_processor.add_keyword(keyword, clean_name)

    # Load data for predicting
    d_list = load_data(config.FEVER_DEV_JSONL)
    sample_answer(d_list, tok, keyword_p=keyword_processor)

    # save the the results for evaluating
    out_fname = config.RESULT_PATH / "doc_retri" / f"{utils.get_current_time_str()}_r" / "dev.jsonl"
    save_intermidiate_results(d_list, out_filename=out_fname)

    # Evaluating
    # out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_06_29_17:41:14_r/dev.jsonl'
    d_list = load_data(out_fname)
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    # print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=Path(out_fname).parent / "analysis.log"))
    print(fever_score(d_list, d_list, mode=eval_mode))


# if token == '-LRB-':
#     return '('
# if token == '-RRB-':
#     return ')'
# if token == '-LSB-':
#     return '['
# if token == '-RSB-':
#     return ']'
# if token == '-LCB-':
#     return '{'
# if token == '-RCB-':
#     return '}'

parentheses_dict = [
    ('-LRB-', '-RRB-'),
    ('-LSB-', '-RSB-'),
    ('-LCB-', '-RCB-')
]


def get_words_inside_parenthese(seq):
    r_list = []
    stacks = [[] for _ in parentheses_dict]

    for t in seq:
        jump_to_next = False
        for i, (l_s, r_s) in enumerate(parentheses_dict):
            if t == l_s:
                stacks[i].append(l_s)
                jump_to_next = True
            elif t == r_s:
                stacks[i].pop()
                jump_to_next = True

        if not jump_to_next and any([len(stack) != 0 for stack in stacks]):
            r_list.append(t)

    return r_list


def check_parentheses(seq):
    stacks = [[] for _ in parentheses_dict]
    for t in seq:
        for i, (l_s, r_s) in enumerate(parentheses_dict):
            if t == l_s:
                stacks[i].append(l_s)
            elif t == r_s:
                if len(stacks[i]) <= 0:
                    # print(seq)
                    return False
                stacks[i].pop()

    valid = True
    for stack in stacks:
        if len(stack) != 0:
            valid = False
            break

    return valid


def remove_parentheses(seq):
    new_seq = []
    stacks = [[] for _ in parentheses_dict]

    for t in seq:
        jump_to_next = False
        for i, (l_s, r_s) in enumerate(parentheses_dict):
            if t == l_s:
                stacks[i].append(l_s)
                jump_to_next = True
            elif t == r_s:
                stacks[i].pop()
                jump_to_next = True

        if not jump_to_next and all([len(stack) == 0 for stack in stacks]):
            new_seq.append(t)

    if new_seq == seq:
        return []

    return new_seq


def id_dict_key_word_expand(id_to_key_dict, create_new_key_word_dict=False):
    """
    :param id_to_key_dict: Original key word dictionary:
    { 'document_id' : [key_trigger_words] }
    :param create_new_key_word_dict: Whether to create a new dictionary or just expand the original one.

    :return: { 'document_id' : [key_trigger_words] } with key word without parentheses in the list.
    """
    if not create_new_key_word_dict:
        for k, v in tqdm(id_to_key_dict.items()):
            if 'disambiguation' in k:   # Removing all disambiguation pages
                continue

            org_keys = copy.deepcopy(v)
            for o_key in org_keys:
                key_t_list = o_key.split(' ')

                if not check_parentheses(key_t_list):
                    print("Pass:", key_t_list)
                    # print()
                else:
                    new_key_t_list = remove_parentheses(key_t_list)
                    if len(new_key_t_list) != 0:
                        id_to_key_dict[k].append(' '.join(new_key_t_list))

            if len(id_to_key_dict[k]) > 1:
                # pass
                # if verbose:
                print(k, id_to_key_dict[k])

        return None
    else:
        new_kw_dict = dict()

        for k, v in tqdm(id_to_key_dict.items()):
            if 'disambiguation' in k:   # Removing all disambiguation pages
                continue

            org_keys = copy.deepcopy(v)
            for o_key in org_keys:
                key_t_list = o_key.split(' ')

                if not check_parentheses(key_t_list):
                    print("Pass:", key_t_list)
                    # print()
                else:
                    new_key_t_list = remove_parentheses(key_t_list)
                    if len(new_key_t_list) != 0:
                        new_kw_dict[k] = [' '.join(new_key_t_list)]
                        # print(k, new_kw_dict[k])

        return new_kw_dict


def used_func_for_fast_key_word_matching_expanded_kw():
    """
    Added on July 1.
    :return:
    """
    # Load tokenizer
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])
    #
    keyword_processor = KeywordProcessor(case_sensitive=True)
    id_to_key_dict = load_keyword_dict(config.DATA_ROOT / "id_dict.jsonl")

    id_dict_key_word_expand(id_to_key_dict, create_new_key_word_dict=False)

    # exit(-2)

    # Write this in a for loop to keep track of the progress
    build_flashtext_processor_wit(keyword_processor, id_to_key_dict)

    # Load data for predicting
    d_list = load_data(config.FEVER_DEV_JSONL)
    sample_answer(d_list, tok, keyword_p=keyword_processor)

    # save the the results for evaluating
    out_fname = config.RESULT_PATH / "doc_retri" / f"{utils.get_current_time_str()}_r" / "dev.jsonl"
    save_intermidiate_results(d_list, out_filename=out_fname)

    # Evaluating
    # out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_07_01_17:20:54_r/dev.jsonl'
    # out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_07_01_17:08:06_r/dev.jsonl'
    # d_list = load_data(out_fname)
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    # print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=Path(out_fname).parent / "analysis.log"))
    print(fever_score(d_list, d_list, mode=eval_mode, verbose=False))


def used_func_for_fast_key_word_matching_prioritized_kw():
    """
    Added on July 1.
    :return:
    """
    # Load tokenizer
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa_yixin.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])
    #
    keyword_processor = KeywordProcessor(case_sensitive=True)

    id_to_key_dict = load_keyword_dict(config.DATA_ROOT / "id_dict.jsonl", filtering=True)

    exact_match_rule_dict = set_priority(id_to_key_dict, priority=5.0)
    print(len(exact_match_rule_dict))

    noisy_key_dict = id_dict_key_word_expand(id_to_key_dict, create_new_key_word_dict=True)
    noisy_parenthese_rule_dict = set_priority(noisy_key_dict, priority=1.0)
    print("Noisy_Parenthese_Rule_Dict:", len(noisy_parenthese_rule_dict))

    # exit(-2)

    # Write this in a for loop to keep track of the progress
    build_flashtext_processor_with_prioritized_kw_dict(keyword_processor, exact_match_rule_dict)
    build_flashtext_processor_with_prioritized_kw_dict(keyword_processor, noisy_parenthese_rule_dict)

    # Load data for predicting
    # d_list = load_data(config.FEVER_TRAIN_JSONL)
    d_list = load_data(config.FEVER_DEV_JSONL)
    sample_answer_with_priority(d_list, tok, keyword_processor, top_k=5)

    # save the the results for evaluating
    # out_fname = config.RESULT_PATH / "doc_retri" / f"{utils.get_current_time_str()}_r" / "train.jsonl"
    out_fname = config.RESULT_PATH / "doc_retri" / f"{utils.get_current_time_str()}_r" / "dev.jsonl"
    save_intermidiate_results(d_list, out_filename=out_fname)

    # Evaluating
    # out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_07_01_17:20:54_r/dev.jsonl'
    # out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_07_01_17:08:06_r/dev.jsonl'
    # d_list = load_data(out_fname)
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    # print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=Path(out_fname).parent / "analysis.log"))
    print(fever_score(d_list, d_list, mode=eval_mode, verbose=False))


def error_analysis(out_fname):
    d_list = load_data(out_fname)
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    # print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=Path(out_fname).parent / "analysis.log"))
    print(fever_score(d_list, d_list, mode=eval_mode,
                      verbose=True, error_analysis_file=Path(out_fname).parent / "analysis.log"))


if __name__ == '__main__':
    # used_func_for_fast_key_word_matching()
    # used_func_for_fast_key_word_matching_expanded_kw()
    used_func_for_fast_key_word_matching_prioritized_kw()
    # error_analysis('/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_18:41:33_r/dev.jsonl')