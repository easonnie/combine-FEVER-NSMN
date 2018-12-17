from utils import common
from utils import c_scorer
from utils import check_sentences
from utils import fever_db
import random


def item_resorting(d_list, top_k=None):
    for item in d_list:
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        # for it in item['prioritized_docids']:
        #     if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
        #         item['predicted_docids'].append(it[0])

        # Reset Exact match
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])

        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]

        # Reset Exact match
        # t_claim = ' '.join(item['claim_tokens'])
        # item['predicted_docids'] = []
        # for k, it in enumerate(item['prioritized_docids']):
        #     if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
        #         item['prioritized_docids'][k] = [it[0], 5.0]
        #         item['predicted_docids'].append(it[0])


def trucate_item(d_list, top_k=None):
    for item in d_list:
        if top_k is not None and len(item['predicted_docids']) > top_k:
            item['predicted_docids'] = item['predicted_docids'][:top_k]


def item_remove_old_rule(d_list):
    for item in d_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                item['prioritized_docids'][i] = [doc_id, 1.0]

        # Reset Exact match
        # t_claim = ' '.join(item['claim_tokens'])
        # item['predicted_docids'] = []
        # for k, it in enumerate(item['prioritized_docids']):
        #     if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
        #         item['prioritized_docids'][k] = [it[0], 5.0]
        #         item['predicted_docids'].append(it[0])


def sample_disamb_inference(d_list, cursor, contain_first_sentence=False):
    inference_list = []
    for item in d_list:
        inference_list.extend(inference_build(item, cursor,
                                              contain_first_sentence=contain_first_sentence))

    return inference_list


def inference_build(item, cursor, contain_first_sentence=False):
    doc_t_list = [it[0] for it in item['prioritized_docids']]
    # evidence_group = check_sentences.check_and_clean_evidence(item)
    t_claim = ' '.join(item['claim_tokens'])
    eid = item['id']

    # This is only for test
    # evidence_group = check_sentences.check_and_clean_evidence(item)
    # all_true_t_list = set()
    # for ground_truth_evid in evidence_group:
    #     print(ground_truth_evid)
    # true_t_list = set([it[0] for it in ground_truth_evid])
    # all_true_t_list = set.union(all_true_t_list, true_t_list)
    # all_true_t_list = list(all_true_t_list)
    # This is only for test

    b_list = []
    for doc_id in doc_t_list:
        if '-LRB-' in doc_id and common.doc_id_to_tokenized_text(doc_id) not in t_claim:

            item = dict()
            item['selection_id'] = str(eid) + '###' + str(doc_id)
            example = common.doc_id_to_tokenized_text(doc_id)
            description_sent = ''
            if contain_first_sentence:
                r_list, id_list = fever_db.get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False)
                for sent, sent_id in zip(r_list, id_list):
                    if int(sent_id.split('(-.-)')[1]) == 0:
                        description_sent = sent

            item['query'] = example + ' ' + description_sent
            item['text'] = t_claim
            item['selection_label'] = 'hidden'

            # if doc_id in all_true_t_list:
            #     item['selection_label'] = 'true'
            # else:
            #     item['selection_label'] = 'false'

            b_list.append(item)

    return b_list


def sample_disamb_training(d_list, cursor, sample_ratio=1.0, contain_first_sentence=False):
    p_list, n_list = [], []
    for item in d_list:
        positive_list, negative_list = disabuigation_training_build(item, cursor,
                                                                    contain_first_sentence=contain_first_sentence)

        if len(positive_list) != 0:
            # print(len(positive_list))
            # print(len(negative_list))
            p_list.extend(positive_list)
            n_list.extend(negative_list)

    # print(len(p_list))
    # print(len(n_list))
    # print(n_list[:10])
    # random.shuffle(n_list)
    n_list = n_list[:int(sample_ratio * len(n_list))]
    r_list = p_list + n_list
    random.shuffle(r_list)

    return p_list + n_list


def sample_disamb_training_v0(d_list, cursor, sample_ratio=1.0, contain_first_sentence=False, only_found=True):
    p_list, n_list = [], []
    for item in d_list:
        positive_list, negative_list = disabuigation_training_build_v0(item, cursor,
                                                                       contain_first_sentence=contain_first_sentence,
                                                                       only_found=only_found)
        # if not relax_pos:
        #     if len(positive_list) != 0:
                # print(len(positive_list))
                # print(len(negative_list))
        p_list.extend(positive_list)
        n_list.extend(negative_list)
        # else:
        #     p_list.extend(positive_list)
        #     n_list.extend(negative_list)

    # print(len(p_list))
    # print(len(n_list))
    # print(n_list[:10])
    # random.shuffle(n_list)
    n_list = n_list[:int(sample_ratio * len(n_list))]
    r_list = p_list + n_list
    random.shuffle(r_list)

    return p_list + n_list


def disabuigation_training_build(item, cursor, contain_first_sentence=False):
    doc_t_list = [it[0] for it in item['prioritized_docids']]
    evidence_group = check_sentences.check_and_clean_evidence(item)
    all_true_t_list = set()
    t_claim = ' '.join(item['claim_tokens'])
    for ground_truth_evid in evidence_group:
        # print(ground_truth_evid)
        true_t_list = set([it[0] for it in ground_truth_evid])
        all_true_t_list = set.union(all_true_t_list, true_t_list)
    all_true_t_list = list(all_true_t_list)

    positive_list = []
    negative_list = []
    eid = item['id']
    found_pos = False

    for doc_id in all_true_t_list:
        if '-LRB-' in doc_id and common.doc_id_to_tokenized_text(doc_id) not in t_claim:
            positive_list.append(doc_id)
            found_pos = True

    if found_pos:
        random.shuffle(doc_t_list)
        num_neg = random.randint(6, 12)

        # for _ in num_neg:
        for doc_id in doc_t_list[:num_neg]:
            if '-LRB-' in doc_id and doc_id not in all_true_t_list:
                negative_list.append(doc_id)

    return make_examples(eid, positive_list, negative_list, t_claim, cursor,
                         contain_first_sentence=contain_first_sentence)


def disabuigation_training_build_v0(item, cursor, contain_first_sentence=False, only_found=True):
    doc_t_list = [it[0] for it in item['prioritized_docids']]
    evidence_group = check_sentences.check_and_clean_evidence(item)
    all_true_t_list = set()
    t_claim = ' '.join(item['claim_tokens'])
    for ground_truth_evid in evidence_group:
        # print(ground_truth_evid)
        true_t_list = set([it[0] for it in ground_truth_evid])
        all_true_t_list = set.union(all_true_t_list, true_t_list)
    all_true_t_list = list(all_true_t_list)

    positive_list = []
    negative_list = []
    eid = item['id']
    found_pos = False

    for doc_id in all_true_t_list:
        if '-LRB-' in doc_id and common.doc_id_to_tokenized_text(doc_id) not in t_claim:
            positive_list.append(doc_id)
            found_pos = True

    if found_pos and only_found:
        random.shuffle(doc_t_list)
        num_neg = random.randint(6, 8)

        # for _ in num_neg:
        for doc_id in doc_t_list[:num_neg]:
            if '-LRB-' in doc_id and doc_id not in all_true_t_list:
                negative_list.append(doc_id)

    elif not only_found:
        random.shuffle(doc_t_list)
        # Change this on Aug 30, 2018
        # num_neg = random.randint(36, 36)
        num_neg = random.randint(6, 8)

        # for _ in num_neg:
        for doc_id in doc_t_list[:num_neg]:
            if '-LRB-' in doc_id and doc_id not in all_true_t_list:
                negative_list.append(doc_id)

    return make_examples(eid, positive_list, negative_list, t_claim, cursor,
                         contain_first_sentence=contain_first_sentence)


def make_examples(eid, positive_list, negative_list, t_claim, cursor, contain_first_sentence=False):
    pos_examples = []
    neg_examples = []

    for pos_e in positive_list:
        item = dict()
        item['selection_id'] = str(eid) + '###' + str(pos_e)
        example = common.doc_id_to_tokenized_text(pos_e)
        description_sent = ''
        if contain_first_sentence:
            r_list, id_list = fever_db.get_all_sent_by_doc_id(cursor, pos_e, with_h_links=False)
            for sent, sent_id in zip(r_list, id_list):
                if int(sent_id.split('(-.-)')[1]) == 0:
                    description_sent = sent

        item['query'] = example + ' ' + description_sent
        item['text'] = t_claim
        item['selection_label'] = 'true'
        pos_examples.append(item)
        del item

    for neg_e in negative_list:
        # sampling
        item = dict()
        item['selection_id'] = str(eid) + '###' + str(neg_e)
        example = common.doc_id_to_tokenized_text(neg_e)
        description_sent = ''
        if contain_first_sentence:
            r_list, id_list = fever_db.get_all_sent_by_doc_id(cursor, neg_e, with_h_links=False)
            for sent, sent_id in zip(r_list, id_list):
                if int(sent_id.split('(-.-)')[1]) == 0:
                    description_sent = sent

        item['query'] = example + ' ' + description_sent
        # print(item['query'])
        item['text'] = t_claim
        item['selection_label'] = 'false'
        neg_examples.append(item)
        del item

    return pos_examples, neg_examples


# Change this method later for flexible usage.
def enforce_disabuigation_into_retrieval_result(disabuigation_r_list, r_list):
    # Index by id and doc_id
    disabuigation_dict = dict()
    for item in disabuigation_r_list:
        disabuigation_dict[item['selection_id']] = item

    for item in r_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                query_id = str(the_id) + '###' + doc_id
                if query_id in disabuigation_dict:
                    query_selection = disabuigation_dict[query_id]
                    # Reset this for flexible usage:
                    if query_selection['selection_label'] == 'true':
                        item['prioritized_docids'][i] = [doc_id, 5.0]
                    if query_selection['selection_label'] == 'false':
                        item['prioritized_docids'][i] = [doc_id, -1.0]


def enforce_disabuigation_into_retrieval_result_v0(disabuigation_r_list, r_list):
    # Index by id and doc_id
    disabuigation_dict = dict()
    for item in disabuigation_r_list:
        disabuigation_dict[item['selection_id']] = item

    for item in r_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                query_id = str(the_id) + '###' + doc_id
                if query_id in disabuigation_dict:
                    query_selection = disabuigation_dict[query_id]
                    # Reset this for flexible usage:
                    # if query_selection['prob'] >= 0.5:
                    #     item['prioritized_docids'][i] = [doc_id, 5.0]
                    # if query_selection['score'] == 'false':
                    #     item['prioritized_docids'][i] = [doc_id, -1.0]
                    item['prioritized_docids'][i] = [doc_id, 1 + query_selection['score'] / 100]

        # Reset Exact match
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])


# with prob
def enforce_disabuigation_into_retrieval_result_v1(disabuigation_r_list, r_list):
    # Index by id and doc_id
    disabuigation_dict = dict()
    for item in disabuigation_r_list:
        disabuigation_dict[item['selection_id']] = item

    for item in r_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                query_id = str(the_id) + '###' + doc_id
                if query_id in disabuigation_dict:
                    query_selection = disabuigation_dict[query_id]
                    # Reset this for flexible usage:
                    # if query_selection['prob'] >= 0.5:
                    #     item['prioritized_docids'][i] = [doc_id, 5.0]
                    # if query_selection['score'] == 'false':
                    #     item['prioritized_docids'][i] = [doc_id, -1.0]
                    # if query_selection['prob']
                    item['prioritized_docids'][i] = [doc_id, 1 + query_selection['prob']]

        # Reset Exact match
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])


# with prob # important we used this function for final selection
def enforce_disabuigation_into_retrieval_result_v2(disabuigation_r_list, r_list, prob_sh=0.5):
    # Index by id and doc_id
    disabuigation_dict = dict()
    for item in disabuigation_r_list:
        disabuigation_dict[item['selection_id']] = item

    for item in r_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:  # Only use for disamb
                query_id = str(the_id) + '###' + doc_id
                if query_id in disabuigation_dict:
                    query_selection = disabuigation_dict[query_id]
                    # Reset this for flexible usage:
                    # if query_selection['prob'] >= 0.5:
                    #     item['prioritized_docids'][i] = [doc_id, 5.0]
                    # if query_selection['score'] == 'false':
                    #     item['prioritized_docids'][i] = [doc_id, -1.0]
                    # if query_selection['prob']
                    item['prioritized_docids'][i] = [doc_id, query_selection['prob']]

        # Reset Exact match
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for k, it in enumerate(item['prioritized_docids']):
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['prioritized_docids'][k] = [it[0], 5.0]
                if it[0] not in item['predicted_docids']:
                    item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids'] and it[1] >= prob_sh:
                item['predicted_docids'].append(it[0])


if __name__ == '__main__':
    d_list = common.load_jsonl("/home/easonnie/projects/FunEver/results/doc_retri_bls/docretri.basic.nopageview/dev.jsonl")
    # d_list = common.load_jsonl("/Users/Eason/RA/FunEver/results/doc_retri_bls/docretri.pageview/dev.jsonl")

    # filtered_list = []
    # for item in d_list:
    #     if filter_contain_parenthese(item):
    # if filter_contain_parenthese_valid(item):
    #     filtered_list.append(item)

    # d_list = filtered_list
    pos_count = 0
    neg_count = 0

    cursor = fever_db.get_cursor()

    p_list, n_list = [], []
    # inference_list = []

    # train_list = sample_disamb_training(d_list, cursor, sample_ratio=1.0)
    # print("Length:", len(train_list))
    # for item in d_list:
    #     positive_list, negative_list = disabuigation_training_build(item, cursor, contain_first_sentence=True)
    #     p_list.extend(positive_list)
    #     n_list.extend(negative_list)

    # for item in d_list:
    #     inference_list.extend(inference_build(item, cursor, contain_first_sentence=False))
    # inference_list = sample_disamb_inference(d_list, cursor)
    train_list = sample_disamb_training_v0(d_list, cursor, only_found=False)
    print(len(d_list))
    print(len(train_list))
    # print(positive_list)
    # print(negative_list)
    # pos_count += len(positive_list)
    # neg_count += len(negative_list)

    # print(len(p_list), len(n_list))

    # item_resorting(d_list)
    # print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))
    #
    # enforce_disabuigation_into_retrieval_result(train_list, d_list)
    # item_resorting(d_list)
    # print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))

    #
    exit(0)
    item_resorting(d_list)
    print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))

    enforce_disabuigation_into_retrieval_result(inference_list, d_list)
    item_resorting(d_list)
    print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))
