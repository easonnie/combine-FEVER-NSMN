import os
from pathlib import Path

SRC_ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
PRO_ROOT = SRC_ROOT.parent

UTEST_ROOT = PRO_ROOT / "utest"
DATA_ROOT = PRO_ROOT / "data"
RESULT_PATH = PRO_ROOT / "results"

WIKI_PAGE_PATH = DATA_ROOT / "wiki-pages"
FEVER_DB = DATA_ROOT / "fever.db"

FEVER_TRAIN_JSONL = DATA_ROOT / "fever" / "train.jsonl"
FEVER_DEV_JSONL = DATA_ROOT / "fever" / "shared_task_dev.jsonl"
FEVER_DEV_UNLABELED_JSONL = DATA_ROOT / "fever" / "shared_task_dev_public.jsonl"

T_FEVER_TRAIN_JSONL = DATA_ROOT / "tokenized_fever" / "train.jsonl"
T_FEVER_DEV_JSONL = DATA_ROOT / "tokenized_fever" / "shared_task_dev.jsonl"

T_FEVER_DT_JSONL = DATA_ROOT / "tokenized_fever" / "dev_train.jsonl"

TOKENIZED_DOC_ID = DATA_ROOT / "tokenized_doc_id.json"

WN_FEATURE_CACHE_PATH = DATA_ROOT / "wn_feature_p"

if __name__ == '__main__':
    print("PRO_ROOT", PRO_ROOT)
    print("SRC_ROOT", SRC_ROOT)
    print("UTEST_ROOT", UTEST_ROOT)
    print("DATA_ROOT", DATA_ROOT)
    print("TOKENIZED_DOC_ID", TOKENIZED_DOC_ID)