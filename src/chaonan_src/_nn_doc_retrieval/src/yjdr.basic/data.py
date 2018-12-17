#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
from time import time

from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, \
                                         ELMoTokenCharactersIndexer

import config
from configuration import config as cfg
import nn_doc_retrieval.disabuigation_training as disamb
from utils import fever_db
from data_util.exvocab import load_vocab_embeddings
from data_util.data_readers.fever_sselection_reader import SSelectorReader
from chaonan_src._utils.doc_utils import read_jsonl


__author__ = ['chaonan99', 'yixin1']


class DocIDCorpus(object):
    def __init__(self, jl_path, train=False):
        self.d_list = read_jsonl(jl_path)
        self.train = train
        self.initialized = False
        self.batch_size = None

    def initialize(self):
        print('Data reader initialization ...')
        self.cursor = fever_db.get_cursor()

        # Prepare Data
        token_indexers = {
            'tokens': \
                SingleIdTokenIndexer(namespace='tokens'),
            'elmo_chars': \
                ELMoTokenCharactersIndexer(namespace='elmo_characters')
        }
        self.fever_data_reader = SSelectorReader(token_indexers=token_indexers,
                                                 lazy=cfg.lazy)

        vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT \
                                                   / 'vocab_cache' \
                                                   / 'nli_basic')
        # THis is important
        ns = 'selection_labels'
        vocab.add_token_to_namespace('true', namespace=ns)
        vocab.add_token_to_namespace('false', namespace=ns)
        vocab.add_token_to_namespace('hidden', namespace=ns)
        vocab.change_token_with_index_to_namespace('hidden', -2, namespace=ns)
        # Label value

        vocab.get_index_to_token_vocabulary(ns)

        self.vocab = vocab
        self.weight_dict = weight_dict
        self.initialized = True

    def get_dp_from_id(self, id):
        if not hasattr(self, 'd_list_dict'):
            self.d_list_dict = {item['id']: item for item in self.d_list}
        return self.d_list_dict[int(id)]

    def resample(self):
        print('Resampling ...')
        start = time()
        if self.train:
            complete_upstream_data = \
                disamb.sample_disamb_training_v0(self.d_list,
                                                 self.cursor,
                                                 cfg.pn_ratio,
                                                 cfg.contain_first_sentence)
        else:
            complete_upstream_data = \
                disamb.sample_disamb_inference(self.d_list, self.cursor,
                    contain_first_sentence=cfg.contain_first_sentence)

        random.shuffle(complete_upstream_data)
        self.instances = self.fever_data_reader.read(complete_upstream_data)
        self.complete_upstream_data = complete_upstream_data
        end = time()
        print('Resampling time:', end - start)

    def initialize_from_exist_corpus(self, corpus):
        assert isinstance(corpus, type(self))
        assert corpus.initialized, 'Exist corpus not initialized!'
        self.cursor = corpus.cursor
        self.fever_data_reader = corpus.fever_data_reader
        self.vocab = corpus.vocab
        self.weight_dict = corpus.weight_dict
        self.initialized = True

    def get_batch(self, batch_size, device_num=-1):
        if not self.initialized:
            self.initialize()

        if self.batch_size is None or self.batch_size != batch_size:
            self.batch_size = batch_size
            biterator = BasicIterator(batch_size=batch_size)
            biterator.index_with(self.vocab)
            self.biterator = biterator

        if self.train or not hasattr(self, 'instances'):
            self.resample()

        return self.biterator(self.instances,
                              shuffle=self.train,
                              num_epochs=1,
                              cuda_device=device_num)

    def index_to_sent(self, ind_list):
        indexer = lambda x: \
                  self.vocab.get_token_from_index(x, namespace='tokens')
        return ' '.join([indexer(x) for x in ind_list if x != 0])

    def __len__(self):
        return len(self.complete_upstream_data) \
               if hasattr(self, 'complete_upstream_data') else 0


def main():
    # toy_dev_file = config.RESULT_PATH \
    #                / 'doc_retri_bls/docretri.basic.nopageview/dev_toy.jsonl'
    train_upstream_file = config.RESULT_PATH \
                   / 'doc_retri_bls/docretri.basic.nopageview/train.jsonl'
    batch_size = 10

    corpus = DocIDCorpus(train_upstream_file, train=True)
    batchifier = corpus.get_batch(batch_size)
    a = next(batchifier)

    for i in range(batch_size):
        p = a['premise']['tokens'][i]
        h = a['hypothesis']['tokens'][i]
        dataid, docid = a['pid'][i].split('###')
        print(corpus.index_to_sent(p.tolist()))
        print(corpus.index_to_sent(h.tolist()))
        print(corpus.get_dp_from_id(dataid)['claim'])
        print(docid)
        print(a['selection_label'][i].tolist())
        print()

    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()