import os
from pathlib import Path
import re
import config
from utils import common


def split_by_line(filename, out_path=None, lines=3):
    filename = Path(filename)
    if out_path is None:
        out_path = filename.parent / f"{filename.name}.splits"

    out_path = Path(out_path)
    out_path.mkdir(parents=False, exist_ok=False)

    with open(filename, encoding='utf-8', mode='r') as in_f:
        file_count = 0
        batch_count = 0
        current_opened_file = None

        for line in in_f:
            if current_opened_file is None:
                current_opened_file = open(os.path.join(out_path, str(out_path.stem) + f".s{file_count}"), encoding='utf-8', mode='w')

            current_opened_file.write(line)
            batch_count += 1
            if batch_count == lines:
                current_opened_file = None
                batch_count = 0
                file_count += 1


def merge_by_line(filename, in_path=None):
    filename = Path(filename)

    if in_path is None:
        in_path = filename.parent / f"{filename.name}.splits"

    in_path = Path(in_path)
    # print(in_path)

    file_list = []
    for sfile in in_path.iterdir():
        # print(sfile)
        pattern = re.compile(r'{0}\.s(\d+)'.format(in_path.stem))
        # print(sfile)
        a = pattern.fullmatch(str(sfile.name))
        if a is None:
            continue
        file_list.append((int(a.group(1)), sfile))

    with open(filename, encoding='utf-8', mode='w') as out_f:
        for _, the_file in sorted(file_list, key=lambda x: x[0]):
            print(the_file)
            with open(the_file, encoding='utf-8', mode='r') as in_f:
                for line in in_f:
                    out_f.write(line)


if __name__ == '__main__':
    dev_d = common.load_jsonl(config.T_FEVER_DEV_JSONL)
    train_d = common.load_jsonl(config.T_FEVER_TRAIN_JSONL)
    dt_d = dev_d + train_d
    common.save_jsonl(dt_d, config.T_FEVER_DT_JSONL)

    # split_by_line("/Users/Eason/RA/FunEver/utest/utest_data/test_rand_data.txt",
    #               out_path="/Users/Eason/RA/FunEver/utest/utest_data/test_rand_data_1.txt.splits")
    #
    # merge_by_line('/Users/Eason/RA/FunEver/utest/utest_data/test_rand_data_1.txt')

    # merge_by_line('/Users/Eason/RA/FunEver/results/sent_retri/2018_07_05_17:17:50_r/train')

    # split_by_line("/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/train.jsonl",
    #               out_path="/Users/Eason/RA/FunEver/results/doc_retri/2018_07_04_21:56:49_r/o_train.splits",
    #               lines=20000)
