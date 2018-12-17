import json
import sqlite3
from collections import Counter
from tqdm import tqdm


class Evidences(object):
    # Evidences is a list of docid and sentences line number
    def __init__(self, evidences):
        evidences_set = set()
        for doc_id, line_num in evidences:
            if doc_id is not None and line_num is not None:
                evidences_set.add((doc_id, line_num))

        evidences_list = sorted(evidences_set, key=lambda x: (x[0], x[1]))
        # print(evidences_list)
        self.evidences_list = evidences_list

    def add_sent(self, sent, ln):
        o_set = set(self.evidences_list)
        o_set.add((sent, ln))
        o_set = sorted(o_set, key=lambda x: (x[0], x[1]))
        self.evidences_list = list(o_set)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Evidences):
            return False

        if len(o.evidences_list) != len(self.evidences_list):
            return False

        is_eq = True
        for o, _e in zip(o.evidences_list, self.evidences_list):
            if o != _e:
                is_eq = False
                break

        return is_eq

    def __hash__(self) -> int:
        hash_str_list = []
        for doc_id, line_num in self.evidences_list:
            hash_str_list.append(f'{doc_id}###{line_num}')
        hash_str = '@'.join(hash_str_list)
        return hash_str.__hash__()

    def __repr__(self):
        return '{Evidences: ' + self.evidences_list.__repr__() + '}'

    def __len__(self):
        return self.evidences_list.__len__()

    def __iter__(self):
        return self.evidences_list.__iter__()


def load_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list


def check_and_clean_evidence(item):
    whole_annotators_evidences = item['evidence']
    # print(evidences)
    evidences_list_set = set()
    for one_annotator_evidences_list in whole_annotators_evidences:
        cleaned_one_annotator_evidences_list = []
        for evidence in one_annotator_evidences_list:
            docid, sent_num = evidence[-2], evidence[-1]
            # print(docid, sent_num)
            cleaned_one_annotator_evidences_list.append((docid, sent_num))

        one_annotator_evidences = Evidences(cleaned_one_annotator_evidences_list)
        evidences_list_set.add(one_annotator_evidences)

    return evidences_list_set


def check_evidence_in_db(cursor, doc_id, line_num):
    key = f'{doc_id}(-.-){line_num}'
    # print("SELECT * FROM sentences WHERE id = \"%s\"" % key)
    cursor.execute("SELECT * FROM sentences WHERE id=?", (key,))
    fetched_data = cursor.fetchone()
    if fetched_data is not None:
        _id, text, h_links, doc_id = fetched_data
    else:
        _id, text, h_links, doc_id = None, None, None, None
    return _id, text, h_links


def check_evidence(item, cursor):
    e_list = check_and_clean_evidence(item)
    print("claim:", item['claim'])
    print("label:", item['label'])
    for i, evidences in enumerate(e_list):
        doc_ids = []
        texts = []
        h_linkss = []
        for docid, line_num in evidences:
            _id, text, h_links = check_evidence_in_db(cursor, docid, line_num)
            doc_ids.append(_id)
            texts.append(text)
            h_linkss.append(h_links)
        print(i, doc_ids, texts, h_linkss)

# The evidence is considered to be correct if there exists a complete list of actual evidence that is a subset of the predicted evidence.

if __name__ == '__main__':
    # d_list = load_data('/Users/Eason/projects/downloaded_repos/fever-baselines/data/fever-data/shared_task_dev.jsonl')
    d_list = load_data('/Users/Eason/projects/downloaded_repos/fever-baselines/data/fever-data/shared_task_dev.jsonl')
    # # print(d_list[0])
    # rand_ind = 12
    # check_and_clean_evidence(d_list[rand_ind])

    # for item in d_list:
    #     e_list = check_and_clean_evidence(item)
    #     print(e_list)
    # evidences_list = [('Cretaceous', 8), ('Cretaceous', 0), ('abc', 9), ('Cretaceous', 8), ('abc', 10)]
    # evidences_list1 = [('Cretaceous', 8), ('Cretaceous', 0), ('abc', 9), ('abc', 10)]
    # e = Evidences(evidences_list)
    # e1 = Evidences(evidences_list1)
    #
    # print(e == e1)
    #
    save_path = '/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db'
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    none_num = 0
    # rand_ind = 191
    # check_evidence(d_list[rand_ind], c)

    evidences_length = []

    count = 0

    for item in tqdm(d_list):
        e_list = check_and_clean_evidence(item)
        if len(e_list) >= 3:
            # print(e_list)
            count += 1

    # print(count)
        print("Claim:", item['claim'])

        for evidences in e_list:
            evidences_length.append(len(evidences))
            print("Evidence:")
            for docid, line_num in evidences:
                _id, text, h_links = check_evidence_in_db(c, docid, line_num)
                # print(docid, line_num)
                # print(_id)
                # print(item[''])
                print(text)

        print("----------------")
                # print(h_links)
                # if _id is None:
                #     none_num += 1
    #
    # print(none_num)
    # print(Counter(evidences_length))


    # _id, text, h_links = check_evidence_in_db(c, 'Apple', '0')
    # print(_id)
    # print(text)
    # print(h_links)
