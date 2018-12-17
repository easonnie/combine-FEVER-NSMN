from utils import common
import config
from utils.fever_db import create_db, save_wiki_pages, create_sent_db, build_sentences_table, check_document_id
from utils.tokenize_fever import tokenized_claim
import fire


def tokenization():
    print("Start tokenizing dev and training set.")
    tokenized_claim(config.FEVER_DEV_JSONL, config.T_FEVER_DEV_JSONL)
    tokenized_claim(config.FEVER_TRAIN_JSONL, config.T_FEVER_TRAIN_JSONL)
    print("Tokenization finished.")


def build_database():
    print("Start building wiki document database. This might take a while.")
    create_db(str(config.FEVER_DB))
    save_wiki_pages(str(config.FEVER_DB))
    create_sent_db(str(config.FEVER_DB))
    build_sentences_table(str(config.FEVER_DB))
    check_document_id(str(config.FEVER_DB))
    print("Wiki document database is ready.")


if __name__ == '__main__':
    fire.Fire()
