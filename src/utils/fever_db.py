import sqlite3
from utils.wiki_term_builder import parse_pages_checks, read_data
from utils import text_clean
from tqdm import tqdm
import config
import json
import re


# Write some standard API for query information

def get_cursor(save_path=str(config.FEVER_DB)):
    conn = sqlite3.connect(save_path)
    cursor = conn.cursor()
    return cursor


def get_evidence(cursor, doc_id, line_num):
    key = f'{doc_id}(-.-){line_num}'
    # print("SELECT * FROM sentences WHERE id = \"%s\"" % key)
    cursor.execute("SELECT * FROM sentences WHERE id=?", (key,))
    fetched_data = cursor.fetchone()
    if fetched_data is not None:
        _id, text, h_links, doc_id = fetched_data
    else:
        _id, text, h_links, doc_id = None, None, None, None
    return _id, text, h_links


def get_all_sent_by_doc_id(cursor, doc_id, with_h_links=False):
    cursor.execute("SELECT * FROM sentences WHERE doc_id=?", (doc_id, ))
    fetched_data = cursor.fetchall()
    r_list = []
    id_list = []
    h_links_list = []
    for id, text, h_links, doc_id in fetched_data:
        # print(id, text, h_li
        # nks, doc_id)
        r_list.append(text)
        id_list.append(id)
        h_links_list.append(json.loads(h_links))

    if with_h_links:
        return r_list, id_list, h_links_list
    else:
        return r_list, id_list
# API Ends


    # convert_special
def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string


def reverse_convert_brc(string):
    string = re.sub('\(', '-LRB-', string)
    string = re.sub('\)', '-RRB-', string)
    string = re.sub('\[', '-LSB-', string)
    string = re.sub(']', '-RSB-', string)
    string = re.sub('{', '-LCB-', string)
    string = re.sub('}', '-RCB-', string)
    string = re.sub(':', '-COLON-', string)
    return string


def create_db(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text, lines_json);")


def create_sent_db(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE sentences (id PRIMARY KEY, text, h_links, doc_id);")
    c.execute("CREATE INDEX doc_id_index ON sentences(doc_id);")


def insert_many(cursor, items):
    cursor.executemany("INSERT INTO documents VALUES (?,?,?)", items)


def save_file_to_db(file, cursor):
    d_list = read_data(file)
    batch_size = 20
    for start in range(0, len(d_list), batch_size):
        end = start + batch_size
        buffer_list = []
        for one_entity in d_list[start:end]:
            this_item = parse_pages_checks(0, one_entity)
            buffer_list.append((this_item['id'], this_item['text'], this_item['lines']))

        insert_many(cursor, buffer_list)
        del buffer_list
        # print(buffer_list)


def save_wiki_pages(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    for file in tqdm(config.WIKI_PAGE_PATH.iterdir()):
        print(file)
        save_file_to_db(file, cursor=c)
        conn.commit()
    conn.close()


def iter_over_db(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("SELECT * from documents")
    count = 0
    for pid, text, lines in tqdm(c, total=5416537):
        pid_words = pid.strip().split('_')
        print(pid_words, len(pid_words))

        if len(text) > 1:
            lines_items = json.loads(lines)
            for line in lines_items:
                # print(line['sentences'])
                if line['sentences']:
                    count += 1

    print(count)


def get_all_doc_ids(save_path, max_ind=None):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("SELECT * from documents")
    id_list = []
    count = 0
    for pid, text, lines in tqdm(c, total=5416537):
        text = text.strip()
        if len(text) > 1 and len(lines) > 0:    # len(text) > 1 will leave out some error documents
            id_list.append(pid)
            count += 1

        if max_ind is not None and count > max_ind:
            break
        # if len(text) == 1:
        #     print(pid, text)

        # print(convert_brc(pid))

    # print(len(id_list))
    # print(len(set(id_list)))
    return id_list


def build_sentences_table(save_path):
    conn = sqlite3.connect(save_path)
    c_doc = conn.cursor()
    c_doc.execute("SELECT * from documents")

    c_sent = conn.cursor()
    count = 0

    for pid, text, lines in tqdm(c_doc, total=5416537):
        if len(text) > 1:
            lines_items = json.loads(lines)
            for line in lines_items:
                # print(line['sentences'])
                if line['sentences']:
                    count += 1
                    sent_pid = pid + '(-.-)' + str(line['line_num'])
                    sent = line['sentences']
                    h_links = json.dumps(line['h_links'])
                    insert_one_sent(c_sent, (sent_pid, sent, h_links, pid))

    print(count)
    conn.commit()
    conn.close()


def insert_one_sent(cursor, item):
    cursor.execute("INSERT INTO sentences VALUES (?,?,?,?)", item)


def check_document_id(save_path):
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("SELECT * from documents")
    count = 0
    for pid, text, lines in tqdm(c, total=5416537):
        pid_words = pid.strip().replace('_', ' ')
        match = re.search('[a-zA-Z]', pid_words)
        if match is None:
            print(pid_words)
        elif text_clean.check_arabic(pid_words):
            print('arabic:', pid_words)
        else:
            count += 1

    print(count)
        # print(pid_words, len(pid_words))
        #
        # if len(text) > 1:
        #     lines_items = json.loads(lines)
        #     for line in lines_items:
        #         print(line['sentences'])
                # if line['sentences']:
                #     count += 1
    #
    # print(count)


def hlink_sanity_check():
    cursor = get_cursor()
    doc_c = get_cursor()
    cursor.execute("SELECT * FROM sentences")
    for id, text, h_links, doc_id in tqdm(cursor, total=25_247_794):
        # print(h_links)

        # print(len(h_links))
        h_links = json.loads(h_links)
        if id == 'Hurlingham_Polo_Association(-.-)1':
            h_links.remove('Asociacion Argentina de Polo')

        if len(h_links) % 2 != 0:
            print(id, text, h_links)

        for i, h_link_doc_id in enumerate(h_links[1::2]):
            h_link_doc_id = reverse_convert_brc(h_link_doc_id.replace(' ', '_'))
            # print(h_link_doc_id)

            doc_c.execute("SELECT * FROM documents where id=(?) COLLATE NOCASE", (h_link_doc_id, ))
            doc = doc_c.fetchall()
            # print(h_link_doc_id, doc)
            if len(doc) == 0:
                print(h_link_doc_id)


if __name__ == '__main__':
    hlink_sanity_check()
    # create_db('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db')
    # save_wiki_pages('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db')
    # iter_over_db('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db')
    # get_all_doc_ids(str(config.FEVER_DB))

    # create_sent_db('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db')
    # build_sentences_table('/Users/Eason/projects/downloaded_repos/fever-baselines/yixin_proj/data/fever.db')
    # check_document_id(str(config.FEVER_DB))

    # print(get_all_sent_by_doc_id(get_cursor(), "Whitesnake", with_h_links=True))
