from collections import Counter

from flashtext import KeywordProcessor
import config
import drqa_yixin.tokenizers
from drqa_yixin.tokenizers import CoreNLPTokenizer
from tqdm import tqdm
import json
from utils.fever_db import get_all_doc_ids, convert_brc
from utils.c_scorer import fever_score
from pathlib import Path
import copy
import utils


def build_keyword_dict(did_list, tokenizer, out_filename):
    # item_list = []
    out_f = open(out_filename, encoding='utf-8', mode='w') if out_filename is not None else None
    for doc_id in tqdm(did_list):
        item = dict()
        item['docid'] = doc_id
        item['keys'], item['lemmas'], item['entities'] = did_to_keys(doc_id, tokenizer=tokenizer)
        # item_list.append(item)

        if out_f is not None:
            out_f.write(json.dumps(item) + '\n')

    out_f.flush()
    out_f.close()


def load_keyword_dict(in_filename):
    id_to_key_dict = dict()
    with open(in_filename, encoding='utf-8', mode='r') as in_f:
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            id_to_key_dict[item['docid']] = item['keys']
            # break

    # print(id_to_key_dict)

    return id_to_key_dict


def build_flashtext_processor(keyword_processor, keyword_dict):
    """
    This method convert keyword dictionary to a flashtext consumeable format and build the keyword_processor
    :param keyword_processor:
    :param keyword_dict:
    { 'document_id' : [key_trigger_words] }

    flashtext consumeable
    { 'key_trigger_word', set(document_id)}
    :return:
    """
    # result_dict = dict()
    for doc_id, kwords in tqdm(keyword_dict.items()):
        for kw in kwords:
            if kw in keyword_processor:
                keyword_processor[kw].add(doc_id)
            else:
                keyword_processor.add_keyword(kw, {doc_id})


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

        # item['predicted_docids'] = [] if len(p_e) == 0 else [es[0] for es in p_e]
        # item['predicted_docids'] = [] if len(p_e) == 0 else p_e[0][0]
        # print(item['predicted_docids'])
        # if item['label'] == 'NOT ENOUGH INFO':
        #     count += 1
    print(num_of_select)
    print(num_of_select.most_common())
    print(count)


def used_func_for_building_normalized_key_word_index_for_docids():
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    did_list = get_all_doc_ids(str(config.FEVER_DB), max_ind=None)

    build_keyword_dict(did_list, tok,
                       config.DATA_ROOT / "id_dict.jsonl")


def used_func_for_fast_key_word_matching():
    # Load tokenizer
    path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    drqa.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
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


def id_dict_key_word_expand(id_to_key_dict):
    for k, v in tqdm(id_to_key_dict.items()):
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


def used_func_for_fast_key_word_matching_expanded_kw():
    """
    Added on July 1.
    :return:
    """
    # Load tokenizer
    # path_stanford_corenlp_full_2017_06_09 = str(config.PRO_ROOT / 'dep_packages/stanford-corenlp-full-2017-06-09/*')
    # drqa.tokenizers.set_default('corenlp_classpath', path_stanford_corenlp_full_2017_06_09)
    # tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])
    #
    # keyword_processor = KeywordProcessor(case_sensitive=True)
    # id_to_key_dict = load_keyword_dict(config.DATA_ROOT / "id_dict.jsonl")
    # id_dict_key_word_expand(id_to_key_dict)

    # exit(-2)

    # Write this in a for loop to keep track of the progress
    # build_flashtext_processor(keyword_processor, id_to_key_dict)
    # for clean_name, keywords in tqdm(id_to_key_dict.items()):
    #     if not isinstance(keywords, list):
    #         raise AttributeError("Value of key {} should be a list".format(clean_name))
    #
    #     for keyword in keywords:
    #         keyword_processor.add_keyword(keyword, clean_name)

    # Load data for predicting
    # d_list = load_data(config.FEVER_DEV_JSONL)
    # sample_answer(d_list, tok, keyword_p=keyword_processor)

    # save the the results for evaluating
    # out_fname = config.RESULT_PATH / "doc_retri" / f"{utils.get_current_time_str()}_r" / "dev.jsonl"
    # save_intermidiate_results(d_list, out_filename=out_fname)

    # Evaluating
    out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_07_01_17:20:54_r/dev.jsonl'
    # out_fname = '/Users/Eason/RA/FunEver/results/doc_retri/2018_07_01_17:08:06_r/dev.jsonl'
    d_list = load_data(out_fname)
    eval_mode = {'check_doc_id_correct': True, 'standard': False}
    # print(fever_score(d_list, d_list, mode=eval_mode, error_analysis_file=Path(out_fname).parent / "analysis.log"))
    print(fever_score(d_list, d_list, mode=eval_mode, verbose=False))


if __name__ == '__main__':
    # used_func_for_fast_key_word_matching()
    used_func_for_fast_key_word_matching_expanded_kw()

    # print(utils.get_current_time_str())
    # drqa.tokenizers.set_default('corenlp_classpath', '/Users/Eason/software/stanford-corenlp-full-2017-06-09/*')
    # tok = CoreNLPTokenizer(annotators=['pos', 'lemma', 'ner'])

    #     "java": ["java_2e", "java programing"],
    # keyword_processor = KeywordProcessor()
    # keyword_dict = {
    #     "java": ["java_2e", "java programing"],
    #     "product management": ["PM", "product manager"]
    # }

    # keyword_processor.add_keywords_from_dict(keyword_dict)
    # print(keyword_processor.extract_keywords('I am a product manager for a java_2e platform'))
    # # output ['product management', 'java']
    # keyword_processor.remove_keyword('java_2e')
    # # you can also remove keywords from a list/ dictionary
    # keyword_processor.remove_keywords_from_dict({"product management": ["PM"]})
    #
    # keyword_processor.remove_keywords_from_list(["java programing"])
    #
    # keyword_processor.extract_keywords('I am a product manager for a java_2e platform')
    # db_path = '/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db'
    # did_list = get_all_doc_ids(db_path, max_ind=None)
    #
    # build_keyword_dict(did_list, tok, '/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/id_dict.jsonl')

    # keyword_processor = KeywordProcessor(case_sensitive=True)
    # id_to_key_dict = load_keyword_dict('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/id_dict.jsonl')

    # keyword_processor.add_keywords_from_dict(id_to_key_dict)
    #

    # for clean_name, keywords in tqdm(id_to_key_dict.items()):
    #     if not isinstance(keywords, list):
    #         raise AttributeError("Value of key {} should be a list".format(clean_name))
    #
    #     for keyword in keywords:
    #         keyword_processor.add_keyword(keyword, clean_name)
    #
    # d_list = load_data('/Users/Eason/projects/downloaded_repos/fever-baselines/data/fever-data/shared_task_dev.jsonl')

    # d_list = load_data('/Users/Eason/projects/downloaded_repos/fever-baselines/data/fever-data/train.jsonl')
    # sample_answer(d_list, tok, keyword_p=keyword_processor)
    #
    # save_intermidiate_results(d_list, out_filename='/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/Jun_20_exact_match_doc_selection_results/results_file_train.jsonl')
    #
    # eval_mode = {'check_doc_id_correct': True, 'standard': False}
    # print(fever_score(d_list, d_list, mode=eval_mode))

    # Error analysis

    # d_list = load_data('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/Jun_20_exact_match_doc_selection_results/results_file_dev.jsonl')
    # eval_mode = {'check_doc_id_correct': True, 'standard': False}
    #
    # print(fever_score(d_list, d_list, mode=eval_mode))

    # Error analysis end
    #
    # keys = keyword_processor.extract_keywords('Fox 2000 Pictures released the film Soul Food.')
    # keys = keyword_processor.extract_keywords('Telemundo is a English-language television network.')
    # print(keys)

    # print(len(did_list))
    # for doc_id in did_list:
    #     print(doc_id, did_to_keys(doc_id, tok))
