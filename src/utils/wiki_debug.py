from utils.wiki_term_builder import *

if __name__ == '__main__':
    with open("/Users/Eason/Desktop/Checking.txt", encoding='utf-8') as in_f:
        s = in_f.read()

    print(s)
    r_list = rule_split(s)
    print(len(r_list))

    lines_b = normalize(s)
    lines_b = '\n' + lines_b
    lines = rule_split(lines_b)

    lines = [line.strip().replace('\n', '') for line in lines]

    if len(lines) == 1 and lines[0] == '':
        lines = ['0']

    string_lines = lines_to_items("What", lines)
    print(string_lines)