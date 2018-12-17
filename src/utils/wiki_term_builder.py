import json
from tqdm import tqdm
import config
import unicodedata
import config


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def read_data(file):
    d_list = []
    with open(file, encoding='utf-8', mode='r') as in_f:
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)
    return d_list


def rule_split(lines_b):
    '''
    Important function!
    This function clean the lines in original preprocessed wiki Pages.
    :param lines_b:
    :return: a list of sentence (responded to each sentence)
    '''
    lines = []

    i = 0
    while True:
        start = f'\n{i}\t'
        end = f'\n{i + 1}\t'
        start_index = lines_b.find(start)
        end_index = lines_b.find(end)

        if end_index == -1:
            extra = f'\n{i + 2}\t'
            extra_1 = f'\n{i + 3}\t'
            extra_2 = f'\n{i + 4}\t'

            # print(lines_b[start_index:])
            lines.append(lines_b[start_index:])
            # print('What?')
            # print(lines_b)
            # print(extra, extra_1)
            # print(lines_b.find(extra))
            # print(lines_b.find(extra_1))
            # print(not (lines_b.find(extra) == -1 and lines_b.find(extra_1) == -1))

            if not (lines_b.find(extra) == -1
                    and lines_b.find(extra_1) == -1
                    and lines_b.find(extra_2) == -1):
                print(lines_b)
                print(extra, extra_1)
                print('Error')
            break

        lines.append(lines_b[start_index:end_index])
        i += 1

    return lines


def lines_to_items(page_id, lines):
    lines_list = []

    for i, line in enumerate(lines):
        line_item = dict()

        line_item_list = line.split('\t')

        line_num = line_item_list[0]
        if not line_num.isdigit():
            print("None digit")
            print(page_id)

            print(lines)
            print(k)
        else:
            line_num = int(line_num)

        if int(line_num) != i:
            print("Line num mismath")
            print(int(line_num), i)
            print(page_id)

            print(k)

        line_item['line_num'] = line_num
        line_item['sentences'] = []
        line_item['h_links'] = []

        if len(line_item_list) <= 1:
            lines_list.append(line_item)
            continue

        sent = line_item_list[1].strip()
        h_links = line_item_list[2:]

        if 'thumb' in h_links:
            h_links = []
        else:
            h_links = list(filter(lambda x: len(x) > 0, h_links))

        line_item['sentences'] = sent
        line_item['h_links'] = h_links
        # print(line_num, sent)
        # print(len(h_links))
        # print(sent)
        # assert sent[-1] == '.'

        if len(h_links) % 2 != 0:
            print(page_id)
            for w in lines:
                print(w)
            print("Term mod 2 != 0")

            print("List:", line_item_list)
            print(line_num, sent)
            print(h_links)
            print()

        lines_list.append(line_item)

    # print(len(lines_list), lines_list[-1]['line_num'] + 1)
    assert len(lines_list) == int(lines_list[-1]['line_num'] + 1)
    string_line_item = json.dumps(lines_list)
    return string_line_item


def parse_pages_checks(k, item):
    this_item = dict()
    page_id = normalize(item['id'].strip())
    text = normalize(item['text'].strip())

    lines_b = normalize(item['lines'])
    lines_b = '\n' + lines_b
    lines = rule_split(lines_b)

    this_item['id'] = page_id
    this_item['text'] = text

    lines = [line.strip().replace('\n', '') for line in lines]

    if len(lines) == 1 and lines[0] == '':
        lines = ['0']

    string_lines = lines_to_items(page_id, lines)

    this_item['lines'] = string_lines

    return this_item


if __name__ == '__main__':
    rand_ind = 8
    WIKI_PAGE_PATH = config.DATA_ROOT / "wiki-pages"
    for file in tqdm(WIKI_PAGE_PATH.iterdir()):
        d_list = read_data(file)
        print(file)
        for k, item in enumerate(d_list):
            parse_pages_checks(k, item)

    # d_list = read_data(config.WIKI_PAGE_PATH / "wiki-045.jsonl")
    # for k, item in tqdm(enumerate(d_list)):
    #     parse_pages_checks(k, item)