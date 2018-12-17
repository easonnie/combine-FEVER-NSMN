from utils import common
from utils import c_scorer
from utils import check_sentences
from utils import fever_db


# def filter_contain_parenthese(item):
#     doc_t_list = [it[0] for it in item['prioritized_docids']]
#     # true_t_list = [it[0] for it in item['evidence']]
#     evidence_group = check_sentences.check_and_clean_evidence(item)
#     claim = ' '.join(item['claim_tokens'])
#     # print(evidence_group)
#     for ground_truth_evid in evidence_group:
#         # print(ground_truth_evid)
#         true_t_list = [it[0] for it in ground_truth_evid]
#         if '-LRB-' not in claim:
#             if any(['-LRB-' in doc_id for doc_id in true_t_list]):
#                 if any(['-LRB-' in doc_id for doc_id in doc_t_list]):
#                     # print("True:", true_t_list)
#                     # print("Pred:", doc_t_list)
#                     return True
#     return False


def filter_contain_parenthese_valid(item):
    doc_t_list = [it[0] for it in item['prioritized_docids']]
    evidence_group = check_sentences.check_and_clean_evidence(item)
    all_true_t_list = set()
    t_claim = ' '.join(item['claim_tokens'])
    for ground_truth_evid in evidence_group:
        # print(ground_truth_evid)
        true_t_list = set([it[0] for it in ground_truth_evid])
        all_true_t_list = set.union(all_true_t_list, true_t_list)
    all_true_t_list = list(all_true_t_list)
    for doc_id in all_true_t_list:
        if '-LRB-' in doc_id and doc_id in doc_t_list and common.doc_id_to_tokenized_text(doc_id) not in t_claim:
            return True

    return False


# Change this method later for flexible usage.
def enforce_disabuigation_into_retrieval_result(disabuigation_r_list, r_list):
    # Index by id and doc_id
    disabuigation_dict = dict()
    for item in disabuigation_r_list:
        disabuigation_dict[item['selection_id']] = item

    for item in r_list:
        the_id = item['id']
        for i, (doc_id, priority) in enumerate(item['prioritized_docids']):
            if '-LRB-' in doc_id:   # Only use for disamb
                query_id = str(the_id) + '###' + doc_id
                if query_id in disabuigation_dict:
                    query_selection = disabuigation_dict[query_id]
                    # Reset this for flexible usage:
                    if query_selection['selection_label'] == 'true':
                        item['prioritized_docids'][i] = [doc_id, 5.0]
                    if query_selection['selection_label'] == 'false':
                        item['prioritized_docids'][i] = [doc_id, -1.0]


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

    for doc_id in all_true_t_list:
        if '-LRB-' in doc_id and common.doc_id_to_tokenized_text(doc_id) not in t_claim:
            positive_list.append(doc_id)

    for doc_id in doc_t_list:
        if '-LRB-' in doc_id and doc_id not in all_true_t_list:
            negative_list.append(doc_id)

    # for doc_id in all_true_t_list:
    #     if '-LRB-' in doc_id and doc_id not in claim:
    #         positive_list.append(doc_id)
    #
    # for doc_id in doc_t_list:
    #     if '-LRB-' in doc_id and doc_id not in all_true_t_list:
    #         negative_list.append(doc_id)

    # print("id:", eid)
    # print("Pos:", positive_list)
    # print("Neg:", negative_list)
    # print("Claim:", t_claim)
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
        print(item['query'])
        item['text'] = t_claim
        item['selection_label'] = 'false'
        neg_examples.append(item)
        del item

    return pos_examples, neg_examples


def item_resorting(d_list):
    for item in d_list:
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for it in item['prioritized_docids']:
            if '-LRB-' in it[0] and common.doc_id_to_tokenized_text(it[0]) in t_claim:
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])


if __name__ == '__main__':
    # d_list = common.load_jsonl("/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/dev.jsonl")
    # print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))
    #
    # d_list = common.load_jsonl("/Users/Eason/RA/FunEver/results/doc_retri/cn_util_Jul17_docretri.singularize/dev.jsonl")
    # print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))
    #
    # d_list = common.load_jsonl("/Users/Eason/RA/FunEver/results/doc_retri/docretri.pageview/dev.jsonl")
    d_list = common.load_jsonl("/Users/Eason/RA/FunEver/results/doc_retri_bls/docretri.basic.nopageview/dev.jsonl")
    # d_list = common.load_jsonl("/Users/Eason/RA/FunEver/results/doc_retri_bls/docretri.pageview/dev.jsonl")

    filtered_list = []
    for item in d_list:
        # if filter_contain_parenthese(item):
        if filter_contain_parenthese_valid(item):
            filtered_list.append(item)

    d_list = filtered_list
    pos_count = 0
    neg_count = 0

    cursor = fever_db.get_cursor()

    p_list, n_list = [], []

    for item in d_list:
        positive_list, negative_list = disabuigation_training_build(item, cursor, contain_first_sentence=True)
        p_list.extend(positive_list)
        n_list.extend(negative_list)
        # print(positive_list)
        # print(negative_list)
        # pos_count += len(positive_list)
        # neg_count += len(negative_list)

    print(len(p_list), len(n_list))
    # print(pos_count, neg_count)
    item_resorting(d_list)
    print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))

    enforce_disabuigation_into_retrieval_result(p_list + n_list, d_list)
    item_resorting(d_list)
    print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=5))

    # for i, item in enumerate(d_list):
    # print("S:", len(item['structured_docids']))
    # print("s:", item['structured_docids'])
    # print("p:", item['prioritized_docids'])
    # print("P:", len(item['prioritized_docids']))
    # print(item['prioritized_docids'])
    # id_list = []
    # for k, v in item['structured_docids'].items():
    #     id_list.extend(v)
    # print(id_list)
    # sorted_docids = sorted(id_list, key=lambda x: -x[1])
    # d_list[i]['predicted_docids'] = [s[0] for s in sorted_docids]

    # sorted_docids = sorted(id_list, key=lambda x: -x[1])
    # d_list[i]['predicted_docids'] = [s[0] for s in sorted_docids]
    # d_list[i]['predicted_docids'] = \
    #     list(set([k for k, v \
    #               in sorted(item['prioritized_docids'],
    #                         key=lambda x: (-x[1], x[0]))]))
    # print(item['predicated_docids'])

    for item in d_list:
        # [it[0] for it in item['prioritized_docids']]
        t_claim = ' '.join(item['claim_tokens'])
        item['predicted_docids'] = []
        for it in item['prioritized_docids']:
            if '-LRB-' in it[0] and it[0] in t_claim:
                item['predicted_docids'].append(it[0])

        for it in sorted(item['prioritized_docids'], key=lambda x: (-x[1], x[0])):
            if it[0] not in item['predicted_docids']:
                item['predicted_docids'].append(it[0])

#         item['predicted_docids'] = item['predicted_docids'][:200]
#
#     print(c_scorer.fever_doc_only(d_list, d_list, max_evidence=200))
# #