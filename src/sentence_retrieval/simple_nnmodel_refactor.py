"""
Author: Yixin Nie
Date:   02 Oct 2018 11:28:58

The file is refactored from 'simple_nnmodel.py' for further exploration of the evidence retrieval
We want to control the relatedness score s.t. they will be more useful for the downstream NLI.
"""
import copy

import torch
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules import Embedding, Elmo
from pathlib import Path
from torch import nn
import numpy as np

import os

import config
from data_util.data_readers.fever_sselection_reader import SSelectorReader
from neural_modules.ema import EMA, load_ema_to_model, save_ema_to_file
from sentence_retrieval.sampler_for_nmodel import get_full_list, post_filter, get_additional_list, \
    get_full_list_from_list_d
from data_util.exvocab import load_vocab_embeddings

from log_util import save_tool
import utils

from flint import torch_util
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from neural_modules import biDafAttn
from sample_for_nli.tf_idf_sample_v1_0 import sample_v1_0, select_sent_for_eval, convert_evidence2scoring_format
from utils import c_scorer, common
from simi_sampler_nli_v0 import simi_sampler


class ESIM(nn.Module):
    # This is ESIM sequence matching model
    # lstm
    def __init__(self, rnn_size_in=(1024 + 300, 1024 + 300), rnn_size_out=(300, 300), max_l=100,
                 mlp_d=300, num_of_class=3, drop_r=0.5, activation_type='relu'):

        super(ESIM, self).__init__()
        self.dropout_layer = nn.Dropout(drop_r)

        self.lstm_1 = nn.LSTM(input_size=rnn_size_in[0], hidden_size=rnn_size_out[0],
                              num_layers=1, bidirectional=True, batch_first=True)

        self.lstm_2 = nn.LSTM(input_size=rnn_size_in[1], hidden_size=rnn_size_out[1],
                              num_layers=1, bidirectional=True, batch_first=True)

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
            raise ValueError("Not a valid activation!")

        self.classifier = nn.Sequential(*[nn.Dropout(drop_r), self.mlp_1, activation, nn.Dropout(drop_r), self.sm])

    def count_params(self):
        total_c = 0
        for param in self.parameters():
            if len(param.size()) == 2:
                d1, d2 = param.size()[0], param.size()[1]
                total_c += d1 * d2
        print("Total count:", total_c)

    def display(self):
        for name, param in self.named_parameters():
            print(name, param.data.size())

    def forward(self, layer1_s1, layer2_s1, l1, layer1_s2, layer2_s2, l2):  # [B, T]

        p_s1 = self.dropout_layer(layer1_s1)
        p_s2 = self.dropout_layer(layer1_s2)

        s1_layer1_out = torch_util.auto_rnn(self.lstm_1, p_s1, l1)
        s2_layer1_out = torch_util.auto_rnn(self.lstm_1, p_s2, l2)

        S = self.bidaf.similarity(s1_layer1_out, l1, s2_layer1_out, l2)
        s1_att, s2_att = self.bidaf.get_both_tile(S, s1_layer1_out, s2_layer1_out)

        s1_coattentioned = torch.cat([s1_layer1_out, s1_att, s1_layer1_out - s1_att,
                                      s1_layer1_out * s1_att], dim=2)

        s2_coattentioned = torch.cat([s2_layer1_out, s2_att, s2_layer1_out - s2_att,
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

        options_file = "http://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "http://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        num_of_elmo = 1

        self.max_l = max_l
        self.elmo_embd_layer = Elmo(options_file, weight_file, num_of_elmo, dropout=0)
        self.esim_layer = ESIM(rnn_size_in, rnn_size_out, max_l, mlp_d, num_of_class, drop_r, activation_type)

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
        # [B, T]
        s_tokens = torch_util.length_truncate(s_tokens, self.max_l)
        s1_glove_embd = self.glove_embd_layer(s_tokens)  # [B, T, D]
        s1_elmo_out = self.elmo_embd_layer(s_elmo_chars)  # [B, T, D]
        s1_elmo_embd = torch_util.length_truncate(s1_elmo_out, self.max_l, is_elmo=True)

        s1_mask, s1_len = torch_util.get_length_and_mask(s_tokens)
        assert torch.equal(s1_elmo_embd['mask'], s1_mask)

        return s1_glove_embd, s1_elmo_embd['elmo_representations'][0], s1_len

    def forward(self, batch):
        s1_tokens = batch['premise']['tokens'].to(next(self.parameters()).device)
        s1_elmo_chars = batch['premise']['elmo_chars'].to(next(self.parameters()).device)

        s2_tokens = batch['hypothesis']['tokens'].to(next(self.parameters()).device)
        s2_elmo_chars = batch['hypothesis']['elmo_chars'].to(next(self.parameters()).device)

        s1_glove_embd, s1_elmo_embd, s1_len = self.raw_input_to_esim_input(s1_tokens, s1_elmo_chars)
        s2_glove_embd, s2_elmo_embd, s2_len = self.raw_input_to_esim_input(s2_tokens, s2_elmo_chars)

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


def hidden_eval(model, data_iter, dev_data_list):
    # select < (-.-) > 0
    # non-select < (-.-) > 1
    # hidden < (-.-) > -2

    with torch.no_grad():
        id2label = {
            0: "true",
            1: "false",
            -2: "hidden"
        }

        print("Evaluating ...")
        model.eval()
        totoal_size = 0

        y_pred_logits_list = []
        y_pred_prob_list = []
        y_id_list = []

        for batch_idx, batch in enumerate(tqdm(data_iter)):
            out = model(batch)
            prob = F.softmax(out, dim=1)

            y = batch['selection_label']
            y_id_list.extend(list(batch['pid']))

            y_pred_logits_list.extend(out[:, 0].tolist())
            y_pred_prob_list.extend(prob[:, 0].tolist())

            totoal_size += y.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_logits_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['selection_id'])
            # Matching id

            dev_data_list[i]['score'] = y_pred_logits_list[i]
            dev_data_list[i]['prob'] = y_pred_prob_list[i]
            # Reset neural set

        print('total_size:', totoal_size)

    return dev_data_list


def eval_for_remaining():
    batch_size = 128
    lazy = True

    # SAVE_PATH = "/home/easonnie/projects/FunEver/saved_models/07-16-11:37:07_simple_nn/i(25000)_epoch(1)_(tra_score:0.8188318831883188|clf_acc:95.67680650034835|pr:0.7394326932693269|rec:0.7282478247824783|f1:0.7337976403219241|loss:0.11368581993118955)"
    SAVE_PATH = config.PRO_ROOT / "saved_models/saved_sselector/i(57167)_epoch(6)_(tra_score:0.8850885088508851|raw_acc:1.0|pr:0.3834395939593578|rec:0.8276327632763276|f1:0.5240763176570098)_epoch"
    # SAVE_PATH = config.PRO_ROOT / "saved_models/07-20-01:35:16_simple_nn_startkp_0.4_de_0.05/i(53810)_epoch(4)_(tra_score:0.8577357735773578|raw_acc:1.0|pr:0.671477147714762|rec:0.7866036603660366|f1:0.7244953493898653)_epoch"
    print("Model From:", SAVE_PATH)
    # dev_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    # dev_upstream_file = config.RESULT_PATH / "doc_retri/cn_util_Jul17_docretri.singularize/dev.jsonl"
    # dev_upstream_file = config.RESULT_PATH / "doc_retri/docretri.pageview/dev.jsonl"
    # dev_upstream_file = config.RESULT_PATH / "doc_retri/docretri.pageview/train.jsonl"
    #
    # SAVE_RESULT_TARGET_FOLDER.mkdir()

    incoming_data_file = config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/dev_s.jsonl"
    incoming_data = common.load_jsonl(incoming_data_file)
    SAVE_RESULT_TARGET_FOLDER = config.RESULT_PATH / "sent_retri_nn/remaining_training_cache"

    # out_file_name = "dev_sent.jsonl"
    out_file_name = "remain_dev_sent.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy)

    # complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True)
    # complete_upstream_dev_data = get_full_list(config.T_FEVER_TRAIN_JSONL, dev_upstream_file, pred=True)
    print("Dev size:", len(incoming_data))
    dev_instances = dev_fever_data_reader.read(incoming_data)

    # Load Vocabulary
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value
    vocab.get_index_to_token_vocabulary('selection_labels')

    print(vocab.get_token_to_index_vocabulary('selection_labels'))
    print(vocab.get_vocab_size('tokens'))

    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300, num_of_class=2)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.display()
    model.to(device)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, incoming_data)

    common.save_jsonl(complete_upstream_dev_data, SAVE_RESULT_TARGET_FOLDER / out_file_name)

    total = 0
    hit = 0

    for item in complete_upstream_dev_data:
        assert item['selection_label'] == 'true'
        if item['prob'] >= 0.5:
            hit += 1
        total += 1

    print(hit, total, hit / total)

    # dev_results_list = score_converter_v0(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
    # dev_results_list = score_converter_v0(config.T_FEVER_TRAIN_JSONL, complete_upstream_dev_data)
    # eval_mode = {'check_sent_id_correct': True, 'standard': True}
    # strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, config.T_FEVER_DEV_JSONL,
    #                                                             mode=eval_mode, verbose=False)
    # strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, config.T_FEVER_TRAIN_JSONL,
    #                                                             mode=eval_mode, verbose=False)
    # total = len(dev_results_list)
    # hit = eval_mode['check_sent_id_correct_hits']
    # tracking_score = hit / total

    # print(f"Dev(fever_score/pr/rec/f1):{strict_score}/{pr}/{rec}/{f1}")
    # print(f"Tracking score:", f"{tracking_score}")


def train_fever_v1():
    num_epoch = 10
    seed = 12
    batch_size = 128
    experiment_name = "simple_nn_startkp_0.4_de_0.05"
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 0.4
    sample_prob_decay = 0.05

    dev_upstream_file = config.RESULT_PATH / "doc_retri/cn_util_Jul17_docretri.singularize/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri/cn_util_Jul17_docretri.singularize/train.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    # dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=False)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value

    vocab.get_index_to_token_vocabulary('selection_labels')

    print(vocab.get_token_to_index_vocabulary('selection_labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)

    model.display()
    model.to(device)

    # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    best_dev = -1
    iteration = 0

    start_lr = 0.0002
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # Resampling
        complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False)
        print("Sample Prob.:", keep_neg_sample_prob)
        filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=keep_neg_sample_prob,
                                          seed=12 + i_epoch)
        # Change the seed to avoid duplicate sample...
        keep_neg_sample_prob -= sample_prob_decay
        if keep_neg_sample_prob <= 0:
            keep_neg_sample_prob = 0.005
        print("Sampled_length:", len(filtered_train_data))
        sampled_train_instances = train_fever_data_reader.read(filtered_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['selection_label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 7:
                mod = 20000
            else:
                mod = 8000

            if iteration % mod == 0:
                eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                dev_results_list = score_converter_v0(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, config.T_FEVER_DEV_JSONL,
                                                                            mode=eval_mode, verbose=False)
                total = len(dev_results_list)
                hit = eval_mode['check_sent_id_correct_hits']
                tracking_score = hit / total

                print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
                print("Strict score:", strict_score)
                print(f"Eval Tracking score:", f"{tracking_score}")

                need_save = False
                if tracking_score > best_dev:
                    best_dev = tracking_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})'
                    )

                    torch.save(model.state_dict(), save_path)

        print("Epoch Evaluation...")
        eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
        complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

        dev_results_list = score_converter_v0(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, config.T_FEVER_DEV_JSONL,
                                                                    mode=eval_mode, verbose=False)
        total = len(dev_results_list)
        hit = eval_mode['check_sent_id_correct_hits']
        tracking_score = hit / total

        print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
        print("Strict score:", strict_score)
        print(f"Eval Tracking score:", f"{tracking_score}")

        if tracking_score > best_dev:
            best_dev = tracking_score

        save_path = os.path.join(
            file_path_prefix,
            f'i({iteration})_epoch({i_epoch})_'
            f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})_epoch'
        )

        torch.save(model.state_dict(), save_path)


def train_fever_v2():
    # train_fever_v1 is the old training script.
    # train_fever_v2 is the new training script created on 02 Oct 2018 11:40:24.
    # Here we keep the negative and positive portion to be consistent.
    num_epoch = 10
    seed = 12
    batch_size = 128
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 1
    top_k_doc = 5

    experiment_name = f"simple_nn_remain_{keep_neg_sample_prob}"
    # sample_prob_decay = 0.05

    dev_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/train_doc.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    # dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=False)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True,
                                               top_k=top_k_doc)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value

    vocab.get_index_to_token_vocabulary('selection_labels')

    print(vocab.get_token_to_index_vocabulary('selection_labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)

    model.display()
    model.to(device)

    cloned_empty_model = copy.deepcopy(model)
    ema: EMA = EMA(parameters=model.named_parameters())

    # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    best_dev = -1
    iteration = 0

    start_lr = 0.0002
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    dev_actual_list = common.load_jsonl(config.T_FEVER_DEV_JSONL)

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # Resampling
        complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False,
                                                     top_k=top_k_doc)

        print("Sample Prob.:", keep_neg_sample_prob)
        filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=keep_neg_sample_prob,
                                          seed=12 + i_epoch)

        # Change the seed to avoid duplicate sample...
        # keep_neg_sample_prob -= sample_prob_decay
        # if keep_neg_sample_prob <= 0:
        #     keep_neg_sample_prob = 0.005
        print("Sampled_length:", len(filtered_train_data))

        sampled_train_instances = train_fever_data_reader.read(filtered_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['selection_label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA
            ema(model.named_parameters())
            iteration += 1

            if i_epoch <= 5:
                mod = 8000
            else:
                mod = 8000

            if iteration % mod == 0:
                eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)

                load_ema_to_model(cloned_empty_model, ema)

                # complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                # Only eval EMA
                complete_upstream_dev_data = hidden_eval(cloned_empty_model, eval_iter, complete_upstream_dev_data)

                dev_results_list = score_converter_v1(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data,
                                                      sent_retri_top_k=5,
                                                      sent_retri_scal_prob=0.5)
                # This is only a wrapper for the simi_sampler

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                for a, b in zip(dev_actual_list, dev_results_list):
                    b['predicted_label'] = a['label']
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list,
                                                                            dev_actual_list,
                                                                            mode=eval_mode, verbose=False)
                tracking_score = strict_score
                print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}")
                print("Strict score:", strict_score)
                print(f"Eval Tracking score:", f"{tracking_score}")

                # need_save = False
                # if tracking_score > best_dev:
                #     best_dev = tracking_score
                need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})_ema'
                    )

                    save_ema_to_file(ema, save_path)
                    # torch.save(model.state_dict(), save_path)

        print("Epoch Evaluation...")
        eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)

        load_ema_to_model(cloned_empty_model, ema)
        # complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)
        complete_upstream_dev_data = hidden_eval(cloned_empty_model, eval_iter, complete_upstream_dev_data)

        dev_results_list = score_converter_v1(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data,
                                              sent_retri_top_k=5,
                                              sent_retri_scal_prob=0.5)

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        for a, b in zip(dev_actual_list, dev_results_list):
            b['predicted_label'] = a['label']
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list,
                                                                    dev_actual_list,
                                                                    mode=eval_mode, verbose=False)
        tracking_score = strict_score
        print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}")
        print("Strict score:", strict_score)
        print(f"Eval Tracking score:", f"{tracking_score}")

        if tracking_score > best_dev:
            best_dev = tracking_score

        save_path = os.path.join(
            file_path_prefix,
            f'i({iteration})_epoch({i_epoch})_'
            f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})_epoch_ema'
        )

        save_ema_to_file(ema, save_path)
        # torch.save(model.state_dict(), save_path)


def eval_and_save_v2(model_path, is_ema, saving_dir, save_train_data=True, prob_thresholds=0.5):
    # This method was modified on 21 NOV 2018
    # for evaluating balanced trained selection model with different threshold value.
    # It will then be used for later verification.

    # Evaluate and Save all the sentence pairs results to be used for downstream verificaion
    # 03 Oct 2018 03:56:40.
    seed = 12
    batch_size = 128
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 1
    top_k_doc = 5

    # sample_prob_decay = 0.05
    dev_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/train_doc.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True,
                                               top_k=top_k_doc)

    complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False,
                                                 top_k=top_k_doc)

    print("Dev size:", len(complete_upstream_dev_data))
    print("Train size:", len(complete_upstream_train_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)
    train_instances = train_fever_data_reader.read(complete_upstream_train_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value

    vocab.get_index_to_token_vocabulary('selection_labels')

    print(vocab.get_token_to_index_vocabulary('selection_labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)

    if not is_ema:
        model.load_state_dict(torch.load(model_path))
    else:
        load_ema_to_model(model, model_path)

    model.display()
    model.to(device)

    dev_actual_list = common.load_jsonl(config.T_FEVER_DEV_JSONL)
    train_actual_list = common.load_jsonl(config.T_FEVER_TRAIN_JSONL)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1)
    train_iter = biterator(train_instances, shuffle=False, num_epochs=1)

    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

    if save_train_data:
        complete_upstream_train_data = hidden_eval(model, train_iter, complete_upstream_train_data)
        common.save_jsonl(complete_upstream_train_data, Path(str(saving_dir)) / "train_sent_scores.jsonl")
        common.save_jsonl(complete_upstream_dev_data, Path(str(saving_dir)) / "dev_sent_pred_scores.jsonl")

    if not isinstance(prob_thresholds, list):
        prob_thresholds = [prob_thresholds]

    for scal_prob in prob_thresholds:
        print("Eval Dev Data prob_threshold:", scal_prob)

        dev_results_list = score_converter_v1(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data,
                                              sent_retri_top_k=5,
                                              sent_retri_scal_prob=scal_prob)
        # This is only a wrapper for the simi_sampler

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        for a, b in zip(dev_actual_list, dev_results_list):
            b['predicted_label'] = a['label']
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list,
                                                                    dev_actual_list,
                                                                    mode=eval_mode, verbose=False)
        tracking_score = strict_score
        print(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
        print("Strict score:", strict_score)
        print(f"Eval Tracking score:", f"{tracking_score}")

    if save_train_data:
        print("Build Train Data")
        train_results_list = score_converter_v1(config.T_FEVER_TRAIN_JSONL, complete_upstream_train_data,
                                                sent_retri_top_k=5,
                                                sent_retri_scal_prob=prob_threshold)

        # This is only a wrapper for the simi_sampler

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        for a, b in zip(train_actual_list, train_results_list):
            b['predicted_label'] = a['label']
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(train_results_list,
                                                                    train_actual_list,
                                                                    mode=eval_mode, verbose=False)
        tracking_score = strict_score
        print(f"Train(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
        print("Strict score:", strict_score)
        print(f"Eval Tracking score:", f"{tracking_score}")


def score_converter_v1(tokenized_file, sent_list, sent_retri_top_k, sent_retri_scal_prob):
    sent_select_results_list_1 = simi_sampler.threshold_sampler_insure_unique(tokenized_file, sent_list,
                                                                              sent_retri_scal_prob,
                                                                              top_n=sent_retri_top_k)
    return sent_select_results_list_1


def score_converter_v0(org_data_file, full_sent_list):
    """
        :param org_data_file:
        :param full_sent_list: append full_sent_score list to evidence of original data file
        :return:
        """
    d_list = common.load_jsonl(org_data_file)
    augmented_dict = dict()
    print("Build selected sentences file:", len(full_sent_list))
    for sent_item in tqdm(full_sent_list):
        selection_id = sent_item['selection_id']  # The id for the current one selection.
        org_id = int(selection_id.split('<##>')[0])
        if org_id in augmented_dict:
            augmented_dict[org_id].append(sent_item)
        else:
            augmented_dict[org_id] = [sent_item]

    for item in d_list:
        if int(item['id']) not in augmented_dict:
            cur_predicted_sentids = []
        else:
            cur_predicted_sentids = []  # formating doc_id + c_score.SENTLINT + line_number
            sents = augmented_dict[int(item['id'])]
            # Modify some mechaism here to selection sentence whether by some score or label
            for sent_i in sents:
                if sent_i['prob'] >= 0.5:
                    cur_predicted_sentids.append((sent_i['sid'], sent_i['score']))
                # del sent_i['prob']

            cur_predicted_sentids = sorted(cur_predicted_sentids, key=lambda x: -x[1])

        item['scored_sentids'] = cur_predicted_sentids
        item['predicted_sentids'] = [sid for sid, _ in item['scored_sentids']][:5]
        item['predicted_evidence'] = convert_evidence2scoring_format(item['predicted_sentids'])
        item['predicted_label'] = item['label']  # give ground truth label

    # Removing all score and prob
    for sent_item in full_sent_list:
        if 'score' in sent_item.keys():
            del sent_item['score']
            del sent_item['prob']

    return d_list


def pipeline_first_sent_selection(org_t_file, upstream_in_file, model_save_path, top_k):
    batch_size = 64
    lazy = True
    SAVE_PATH = model_save_path
    print("Model From:", SAVE_PATH)

    dev_upstream_file = upstream_in_file

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy)

    complete_upstream_dev_data = get_full_list(org_t_file, dev_upstream_file, pred=True, top_k=top_k)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value
    vocab.get_index_to_token_vocabulary('selection_labels')

    print(vocab.get_token_to_index_vocabulary('selection_labels'))
    print(vocab.get_vocab_size('tokens'))

    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300, num_of_class=2)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.display()
    model.to(device)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    dev_sent_full_list = hidden_eval(model, eval_iter, complete_upstream_dev_data)

    return dev_sent_full_list


def pipeline_first_sent_selection_list(org_t_file, upstream_in_file, model_save_path, top_k):
    batch_size = 64
    lazy = True
    SAVE_PATH = model_save_path
    print("Model From:", SAVE_PATH)

    dev_upstream_file = upstream_in_file

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy)

    complete_upstream_dev_data = get_full_list_from_list_d(org_t_file, dev_upstream_file, pred=True, top_k=top_k)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value
    vocab.get_index_to_token_vocabulary('selection_labels')

    print(vocab.get_token_to_index_vocabulary('selection_labels'))
    print(vocab.get_vocab_size('tokens'))

    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300, num_of_class=2)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.display()
    model.to(device)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    dev_sent_full_list = hidden_eval(model, eval_iter, complete_upstream_dev_data)

    return dev_sent_full_list


def ana_neg_pos_portion():
    dev_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/train_doc.jsonl"

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True, top_k=5)
    complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False, top_k=5)

    print(len(complete_upstream_dev_data))
    filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=1,
                                      seed=12)

    pos = 0
    neg = 0
    for item in filtered_train_data:
        if item['selection_label'] == 'true':
            pos += 1
        elif item['selection_label'] == 'false':
            neg += 1

    print(pos, neg)

    # print(len(complete_upstream_dev_data))
    print(len(filtered_train_data))


if __name__ == "__main__":
    # ana_neg_pos_portion()
    # train_fever_v2()

    eval_and_save_v2(
        "/home/easonnie/projects/FunEver/saved_models/10-03-20:29:07_simple_nn_remain_1/i(56000)_epoch(1)_(tra_score:0.7957295729572957|raw_acc:1.0|pr:0.86442769276928|rec:0.6935943594359436|f1:0.769645296093697)_ema",
        True,
        "/home/easonnie/projects/FunEver/results/sent_retri_nn/balanced_sentence_selection_results",
        False,
        [0.1, 0.08, 0.06, 0.04, 0.02]
    )