#!/usr/bin/env python
"""
"""

import os
import datetime
import gzip
import pickle
from collections import defaultdict
from glob import glob
from time import time

from chaonan_src._utils.spcl import spcl
import config

__all__ = ['pageview_download', 'count_pageviews']
__author__ = ['chaonan99']


save_path_root = "/playpen2/home/.chaonan99/data/wikimedia/"
link_path_root = "https://dumps.wikimedia.org/other/"

def get_file_name(month, day, hour):
    return f"pageviews/2018/2018-{month:02d}/pageviews-2018" \
           f"{month:02d}{day:02d}-{hour:02d}0000.gz"


def pageview_download():
    import wget

    qtime = datetime.datetime.now()
    one_hour = datetime.timedelta(hours=1)
    qtime -= one_hour
    for _ in range(30*24):
        qtime -= one_hour
        file_name = get_file_name(qtime.month, qtime.day, qtime.hour)
        rlink = link_path_root + file_name
        local_path = save_path_root + file_name

        dir_path = os.path.dirname(local_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        print(f"Downloading {file_name}")
        wget.download(rlink, out=local_path)


def count_pageviews():
    """Count pageviews and save to a dictionary
    """
    fname = save_path_root + get_file_name(7, 19, 12)
    en_wiki_count = defaultdict(lambda: 0)

    all_files = glob(save_path_root + '**/*.gz', recursive=True)
    for fname in spcl(all_files, every=1):
        with gzip.open(fname, mode='rb') as f:
            for l in f:
                if l.startswith(b'en'):
                    lsplits = l.split()
                    if len(lsplits) != 4:
                        continue
                    _, docid, count, _ = lsplits
                    docid = docid.decode('utf-8')
                    count = int(count)
                    en_wiki_count[docid] += count

    print("All done! Cheers!")
    pv_dump_path = str(config.RESULT_PATH) + "/chaonan99/pageviews.pkl"
    pickle.dump(dict(en_wiki_count), open(pv_dump_path, 'wb'))
    from IPython import embed; embed();


class WikiPageviews(object):
    """WikiPageviews"""
    pageview_path = str(config.RESULT_PATH) + "/chaonan99/pageviews.pkl"

    def __init__(self):
        pv_dict_raw = pickle.load(open(self.pageview_path, 'rb'))
        self.pageview_dict = defaultdict(lambda: 0, pv_dict_raw)

    def __getitem__(self, ind):
        return self.pageview_dict[ind]


def pageview_analysis():
    from chaonan_src._doc_retrieval.item_rules import ItemRuleBuilder
    from chaonan_src._utils.doc_utils import read_jsonl
    from utils.fever_db import convert_brc

    wiki_pv = WikiPageviews()
    d_list = read_jsonl("../../../results/doc_retri/docretri.titlematch/dev.jsonl")
    gt_evidences, pre_evidences = [], []
    for item in d_list:
        gt_evidences.extend(ItemRuleBuilder\
            .get_all_docid_in_evidence(item['evidence']))
        pre_evidences.extend([it[0] for it in item['prioritized_docids']])
    gt_evidences = set(gt_evidences)
    pre_evidences = set(pre_evidences)

    gt_count = [wiki_pv[convert_brc(it)] for it in gt_evidences]
    pre_count = [wiki_pv[convert_brc(it)] for it in pre_evidences]


    from IPython import embed; embed(); import os; os._exit(1)


def main():
    wiki_pv = WikiPageviews()
    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()