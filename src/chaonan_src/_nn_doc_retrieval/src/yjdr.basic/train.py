#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import os

import torch
import torch.optim as optim

import config
import nn_doc_retrieval.disabuigation_training as disamb
from utils import c_scorer
from configuration import config as cfg
from data import DocIDCorpus
from model import Model
from predict import hidden_eval


__author__ = ['chaonan99', 'yixin1']


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        self.initialize()

    def initialize(self):
        self.logger = cfg.get_logger()
        self.info(repr(cfg))

        device = torch.device('cuda' if torch.cuda.is_available() \
                                         else 'cpu', index=0)
        device_num = -1 if device.type == 'cpu' else 0
        self.info(f'Use {device}')

        ########################################################################
        # Load data
        ########################################################################
        dev_upstream_file = config.RESULT_PATH / \
                        "doc_retri_bls/docretri.basic.nopageview/dev.jsonl"
        train_upstream_file = config.RESULT_PATH / \
                        "doc_retri_bls/docretri.basic.nopageview/train.jsonl"
        dev_corpus = DocIDCorpus(dev_upstream_file, train=False)
        train_corpus = DocIDCorpus(train_upstream_file, train=True)
        dev_corpus.initialize()
        train_corpus.initialize_from_exist_corpus(dev_corpus)

        ########################################################################
        # Build the model
        ########################################################################
        model = Model(weight=dev_corpus.weight_dict['glove.840B.300d'],
                      vocab_size=dev_corpus.vocab.get_vocab_size('tokens'),
                      embedding_dim=300, max_l=160, num_of_class=2)
        model.to(device)
        criterion = model.get_criterion()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()),
                               lr=cfg.start_lr)

        self.device_num = device_num
        self.train_corpus = train_corpus
        self.dev_corpus = dev_corpus
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_dev = -1

    def info(self, info_str):
        self.logger.info(info_str)

    def train_epoch(self):
        """One epoch training
        """
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        epoch = self.epoch
        train_corpus = self.train_corpus
        device_num = self.device_num

        model.train()
        total_loss = 0.
        self.start_time = time.time()
        batchfier = train_corpus.get_batch(cfg.batch_size,
                                           device_num=device_num)
        total_iter = len(train_corpus) // cfg.batch_size

        for batch, data_batch in enumerate(batchfier):
            out = model(data_batch)
            y = data_batch['selection_label']
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch % cfg.log_interval == 0 and batch > 0:
                cur_loss = total_loss / cfg.log_interval
                elapsed = time.time() - self.start_time
                cur_lr = optimizer.param_groups[0]['lr']
                self.info(
                    f'| epoch {epoch:3d} | '
                    f'{batch:5d}/{total_iter:5d} batches | '
                    f'lr {cur_lr:02.2e} | '
                    f'ms/batch {elapsed * 1000 / cfg.log_interval:5.2f} | '
                    f'loss {cur_loss:2.5f} |')
                total_loss = 0
                self.start_time = time.time()

    def train(self):
        try:
            for self.epoch in range(cfg.num_epoch):
                self.train_epoch()
                self.evaluate(self.dev_corpus)
        except KeyboardInterrupt:
            self.info('-' * 89)
            self.info('Exiting from training early. Save current model.')
            self.save_model('quit_dump')

    def evaluate(self, corpus):
        model = self.model
        device_num = self.device_num

        eval_iter = corpus.get_batch(cfg.dev_batch_size, device_num=device_num)
        complete_upstream_data = hidden_eval(model,
                                             eval_iter,
                                             corpus.complete_upstream_data)

        disamb.enforce_disabuigation_into_retrieval_result_v0( \
            complete_upstream_data, corpus.d_list)
        oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(corpus.d_list,
                                                            corpus.d_list,
                                                            max_evidence=5)
        elapsed = time.time() - self.start_time
        self.info(
            f'| epoch {self.epoch:3d} | '
            f'oracle_score {oracle_score:5.6f} | '
            f'pr {pr:5.2f} | '
            f'rec {rec:5.2f} | '
            f'f1 {f1:5.2f} | '
            f'eval time {elapsed:6.2f} s |'
            )
        if oracle_score > self.best_dev:
            self.best_dev = oracle_score
            self.save_model(f'epoch({self.epoch})_'
                            f'(oracle_score:{oracle_score}'
                            f'|pr:{pr}|rec:{rec}|f1:{f1})')

        self.start_time = time.time()
        return oracle_score

    def save_model(self, model_name='model'):
        save_path = os.path.join(cfg.save_path, f'{model_name}.pt')
        torch.save(self.model.state_dict(), save_path)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()