#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch import nn
from allennlp.modules import Embedding, Elmo

from flint import torch_util
from neural_modules import biDafAttn
from configuration import config as cfg


__author__ = ['chaonan99', 'yixin1']


class ESIM(nn.Module):
    # This is ESIM sequence matching model
    # lstm
    def __init__(self,
                 rnn_size_in=(1024 + 300, 1024 + 300),
                 rnn_size_out=(300, 300),
                 max_l=100,
                 mlp_d=300,
                 num_of_class=3,
                 drop_r=0.5,
                 activation_type='relu'):

        super(ESIM, self).__init__()
        self.dropout_layer = nn.Dropout(drop_r)

        self.lstm_1 = nn.LSTM(input_size=rnn_size_in[0],
                              hidden_size=rnn_size_out[0],
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)

        self.lstm_2 = nn.LSTM(input_size=rnn_size_in[1],
                              hidden_size=rnn_size_out[1],
                              num_layers=1,
                              bidirectional=True,
                              batch_first=True)

        self.projection = nn.Linear(rnn_size_out[0] * 2 * 4, rnn_size_out[0])

        self.max_l = max_l
        self.bidaf = biDafAttn(300)

        self.mlp_1 = nn.Linear(rnn_size_out[1] * 2 * 4, mlp_d)
        self.sm = nn.Linear(mlp_d, num_of_class)

        if activation_type == 'relu':
            activation = nn.ReLU()
        elif activation_type == 'tanh':
            activation = nn.Tanh()
        else:
            raise ValueError('Not a valid activation!')

        self.classifier = nn.Sequential(*[nn.Dropout(drop_r),
                                          self.mlp_1,
                                          activation,
                                          nn.Dropout(drop_r),
                                          self.sm])

    def count_params(self):
        total_c = 0
        for param in self.parameters():
            if len(param.size()) == 2:
                d1, d2 = param.size()[0], param.size()[1]
                total_c += d1 * d2
        print('Total count:', total_c)

    def display(self):
        for name, param in self.named_parameters():
            print(name, param.data.size())

    def forward(self, layer1_s1, layer2_s1, l1, layer1_s2, layer2_s2, l2):

        p_s1 = self.dropout_layer(layer1_s1)
        p_s2 = self.dropout_layer(layer1_s2)

        s1_layer1_out = torch_util.auto_rnn(self.lstm_1, p_s1, l1)
        s2_layer1_out = torch_util.auto_rnn(self.lstm_1, p_s2, l2)

        S = self.bidaf.similarity(s1_layer1_out, l1, s2_layer1_out, l2)
        s1_att, s2_att = self.bidaf.get_both_tile(S,
                                                  s1_layer1_out,
                                                  s2_layer1_out)

        s1_coattentioned = torch.cat([s1_layer1_out, s1_att,
                                      s1_layer1_out - s1_att,
                                      s1_layer1_out * s1_att], dim=2)

        s2_coattentioned = torch.cat([s2_layer1_out, s2_att,
                                      s2_layer1_out - s2_att,
                                      s2_layer1_out * s2_att], dim=2)

        p_s1_coattentioned = F.relu(self.projection(s1_coattentioned))
        p_s2_coattentioned = F.relu(self.projection(s2_coattentioned))

        s1_coatt_features = torch.cat([p_s1_coattentioned, layer2_s1], dim=2)
        s2_coatt_features = torch.cat([p_s2_coattentioned, layer2_s2], dim=2)

        s1_coatt_features = self.dropout_layer(s1_coatt_features)
        s2_coatt_features = self.dropout_layer(s2_coatt_features)

        s1_layer2_out = torch_util.auto_rnn(self.lstm_2, s1_coatt_features, l1)
        s2_layer2_out = torch_util.auto_rnn(self.lstm_2, s2_coatt_features, l2)

        s1_lay2_maxout = torch_util.max_along_time(s1_layer2_out, l1)
        s2_lay2_maxout = torch_util.max_along_time(s2_layer2_out, l2)

        features = torch.cat([s1_lay2_maxout, s2_lay2_maxout,
                              torch.abs(s1_lay2_maxout - s2_lay2_maxout),
                              s1_lay2_maxout * s2_lay2_maxout], dim=1)

        return self.classifier(features)


class Model(nn.Module):
    def __init__(self, weight, vocab_size, embedding_dim,
                 rnn_size_in=(1024 + 300, 1024 + 300),
                 rnn_size_out=(300, 300), max_l=150,
                 mlp_d=300, num_of_class=3, drop_r=0.5, activation_type='relu'):

        super(Model, self).__init__()
        self.glove_embd_layer = Embedding(vocab_size, embedding_dim,
                                          weight=weight, padding_index=0)

        options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/'\
                       'elmo/2x4096_512_2048cnn_2xhighway/'\
                       'elmo_2x4096_512_2048cnn_2xhighway_options.json'
        weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/'\
                      'elmo/2x4096_512_2048cnn_2xhighway/'\
                      'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
        num_of_elmo = 1

        self.max_l = max_l
        self.elmo_embd_layer = Elmo(options_file,
                                    weight_file,
                                    num_of_elmo,
                                    dropout=0)
        self.esim_layer = ESIM(rnn_size_in,
                               rnn_size_out,
                               max_l,
                               mlp_d,
                               num_of_class,
                               drop_r,
                               activation_type)

    def display(self, exclude=None):
        total_p_size = 0
        if exclude is None:
            exclude = {'glove'}

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.size())

                exclude_this = False
                for exclude_name in exclude:
                    if exclude_name in str(name):
                        exclude_this = True

                if exclude_this:
                    continue

                nn = 1
                for s in list(param.size()):
                    nn = nn * s
                total_p_size += nn

        print('Total Size:', total_p_size)

    def raw_input_to_esim_input(self, s_tokens, s_elmo_chars):
        s_tokens = torch_util.length_truncate(s_tokens, self.max_l)
        s1_glove_embd = self.glove_embd_layer(s_tokens)
        s1_elmo_out = self.elmo_embd_layer(s_elmo_chars)
        s1_elmo_embd = torch_util.length_truncate(s1_elmo_out,
                                                  self.max_l,
                                                  is_elmo=True)

        s1_mask, s1_len = torch_util.get_length_and_mask(s_tokens)
        assert torch.equal(s1_elmo_embd['mask'], s1_mask)

        return s1_glove_embd, s1_elmo_embd['elmo_representations'][0], s1_len

    def forward(self, batch):
        s1_tokens = batch['premise']['tokens']
        s1_elmo_chars = batch['premise']['elmo_chars']

        s2_tokens = batch['hypothesis']['tokens']
        s2_elmo_chars = batch['hypothesis']['elmo_chars']

        s1_glove_embd, s1_elmo_embd, s1_len = \
            self.raw_input_to_esim_input(s1_tokens, s1_elmo_chars)
        s2_glove_embd, s2_elmo_embd, s2_len = \
            self.raw_input_to_esim_input(s2_tokens, s2_elmo_chars)

        s1_layer1_in = torch.cat((s1_glove_embd, s1_elmo_embd), dim=2)
        s1_layer2_in = s1_elmo_embd

        s2_layer1_in = torch.cat((s2_glove_embd, s2_elmo_embd), dim=2)
        s2_layer2_in = s2_elmo_embd

        # print(s1_layer1_in.size())
        # print(s1_layer2_in.size())
        # print(s2_layer1_in.size())
        # print(s2_layer2_in.size())
        esim_out = self.esim_layer(s1_layer1_in, s1_layer2_in, s1_len,
                                   s2_layer1_in, s2_layer2_in, s2_len)

        return esim_out

    @classmethod
    def get_criterion(cls):
        return nn.CrossEntropyLoss()


def main():
    from data import DocIDCorpus
    import config

    # dev_upstream_file = config.RESULT_PATH \
    #                     / 'doc_retri_bls/docretri.basic.nopageview/dev.jsonl'
    train_upstream_file = config.RESULT_PATH \
                        / 'doc_retri_bls/docretri.basic.nopageview/train.jsonl'
    device = torch.device('cuda' if torch.cuda.is_available() \
                                 else 'cpu', index=0)
    device_num = -1 if device.type == 'cpu' else 0

    corpus = DocIDCorpus(train_upstream_file, train=True)
    # batchifier = corpus.get_batch(10)
    batchifier = corpus.get_batch(100, device_num)

    model = Model(weight=corpus.weight_dict['glove.840B.300d'],
                  vocab_size=corpus.vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)
    model.display()
    model.to(device)
    criterion = model.get_criterion()

    # for i, batch in enumerate(batchifier):
    #     print(i, end='', flush=True)
    #     if batch['selection_label'].sum().tolist() > 1:
    #         print(batch)
    #         break

    batch = next(batchifier)
    out = model(batch)
    y = batch['selection_label']
    loss = criterion(out, y)

    print(loss.item())

    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()