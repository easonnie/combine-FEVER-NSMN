import random

import torch
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules import Embedding, Elmo
from torch import nn
from utils import fever_db
import numpy as np

import os

import config
from data_util.data_readers.fever_sselection_reader import SSelectorReader
import nn_doc_retrieval.disabuigation_training as disamb
from sentence_retrieval.sampler_for_nmodel import get_full_list, post_filter, get_additional_list
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
        s_tokens = torch_util.length_truncate(s_tokens, self.max_l)
        s1_glove_embd = self.glove_embd_layer(s_tokens)
        s1_elmo_out = self.elmo_embd_layer(s_elmo_chars)
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


def eval_model(model, data_iter, criterion):
    print("Evaluating ...")
    model.eval()
    n_correct = loss = 0
    totoal_size = 0

    y_pred_list = []
    y_true_list = []

    for batch_idx, batch in enumerate(data_iter):
        out = model(batch)
        y = batch['label']

        n_correct += (torch.max(out, 1)[1].view(y.size()) == y).sum().item()

        y_pred_list.extend(torch.max(out, 1)[1].view(y.size()).tolist())
        y_true_list.extend(y.tolist())

        loss += criterion(out, y).item() * y.size(0)
        totoal_size += y.size(0)

    print('n_correct:', n_correct)
    print('total_size:', totoal_size)

    avg_acc = 100. * n_correct / totoal_size
    avg_loss = loss / totoal_size

    return avg_acc, avg_loss


def full_eval_model(model, data_iter, criterion, dev_data_list):
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
        n_correct = loss = 0
        totoal_size = 0

        y_pred_logits_list = []
        y_pred_prob_list = []
        y_id_list = []

        for batch_idx, batch in enumerate(tqdm(data_iter)):
            out = model(batch)
            prob = F.softmax(out, dim=1)

            y = batch['selection_label']
            y_id_list.extend(list(batch['pid']))

            n_correct += (torch.max(out, 1)[1].view(y.size()) == y).sum().item()
            y_pred_logits_list.extend(out[:, 0].tolist())
            y_pred_prob_list.extend(prob[:, 0].tolist())

            loss += criterion(out, y).item() * y.size(0)
            totoal_size += y.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_logits_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['selection_id'])
            # Matching id

            dev_data_list[i]['score'] = y_pred_logits_list[i]
            dev_data_list[i]['prob'] = y_pred_prob_list[i]
            # Reset neural set

        print('n_correct:', n_correct)
        print('total_size:', totoal_size)

        avg_acc = 100. * n_correct / totoal_size
        avg_loss = loss / totoal_size

    return avg_acc, avg_loss, dev_data_list


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


def train_fever():
    num_epoch = 8
    seed = 12
    batch_size = 128
    experiment_name = "simple_nn"
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 0.5
    sample_prob_decay = 0.1

    dev_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/train.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy)
    # dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=False)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy)

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="labels")
    vocab.add_token_to_namespace("false", namespace="labels")
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')
    # Label value

    vocab.get_index_to_token_vocabulary('labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    dev_biterator.index_with(vocab)

    # exit(0)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300, num_of_class=2)

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
        filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=keep_neg_sample_prob,
                                          seed=12 + i_epoch)
        # Change the seed to avoid duplicate sample...
        keep_neg_sample_prob -= sample_prob_decay

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

            if i_epoch <= 4:
                mod = 25000
            else:
                mod = 10000

            if iteration % mod == 0:
                eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss, complete_upstream_dev_data = full_eval_model(model, eval_iter, criterion,
                                                                                  complete_upstream_dev_data)

                dev_results_list = score_converter_v0(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, config.T_FEVER_DEV_JSONL,
                                                                            mode=eval_mode, verbose=False)
                total = len(dev_results_list)
                hit = eval_mode['check_sent_id_correct_hits']
                tracking_score = hit / total

                print(f"Dev(clf_acc/pr/rec/f1/loss):{dev_score}/{pr}/{rec}/{f1}/{dev_loss}")
                print(f"Tracking score:", f"{tracking_score}")

                need_save = False
                if tracking_score > best_dev:
                    best_dev = tracking_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{tracking_score}|clf_acc:{dev_score}|pr:{pr}|rec:{rec}|f1:{f1}|loss:{dev_loss})'
                    )

                    torch.save(model.state_dict(), save_path)


def train_fever_v1():
    num_epoch = 10
    seed = 12
    batch_size = 64
    dev_batch_size = 128
    experiment_name = "simple_nn_doc_first_sent"
    # experiment_name = "simple_nn_doc"
    lazy = True
    torch.manual_seed(seed)
    contain_first_sentence = True
    pn_ratio = 1.0
    # keep_neg_sample_prob = 0.4
    # sample_prob_decay = 0.05

    dev_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/train.jsonl"
    dev_data_list = common.load_jsonl(dev_upstream_file)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    # dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=False)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    cursor = fever_db.get_cursor()
    complete_upstream_dev_data = disamb.sample_disamb_inference(common.load_jsonl(dev_upstream_file), cursor,
                                                                contain_first_sentence=contain_first_sentence)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=dev_batch_size)

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
        complete_upstream_train_data = disamb.sample_disamb_training_v0(common.load_jsonl(train_upstream_file),
                                                                        cursor, pn_ratio, contain_first_sentence,
                                                                        only_found=False)
        random.shuffle(complete_upstream_train_data)
        print("Sample Prob.:", pn_ratio)

        print("Sampled_length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

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

            if i_epoch <= 5:
                mod = 1000
            else:
                mod = 500

            if iteration % mod == 0:
                eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                disamb.enforce_disabuigation_into_retrieval_result_v0(complete_upstream_dev_data,
                                                                      dev_data_list)
                oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

                print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
                print("Strict score:", oracle_score)
                print(f"Eval Tracking score:", f"{oracle_score}")

                need_save = False
                if oracle_score > best_dev:
                    best_dev = oracle_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{oracle_score}|pr:{pr}|rec:{rec}|f1:{f1})'
                    )

                    torch.save(model.state_dict(), save_path)
        # 
        print("Epoch Evaluation...")
        eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
        complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

        disamb.enforce_disabuigation_into_retrieval_result_v0(complete_upstream_dev_data,
                                                              dev_data_list)
        oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

        print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
        print("Strict score:", oracle_score)
        print(f"Eval Tracking score:", f"{oracle_score}")

        need_save = False
        if oracle_score > best_dev:
            best_dev = oracle_score
            need_save = True

        if need_save:
            save_path = os.path.join(
                file_path_prefix,
                f'i({iteration})_epoch({i_epoch})_e'
                f'(tra_score:{oracle_score}|pr:{pr}|rec:{rec}|f1:{f1})'
            )

            torch.save(model.state_dict(), save_path)


def doc_model_eval():
    seed = 12
    batch_size = 128
    dev_batch_size = 128
    lazy = True
    torch.manual_seed(seed)
    contain_first_sentence = True
    # keep_neg_sample_prob = 0.4
    # sample_prob_decay = 0.05

    # model_path = "/home/easonnie/projects/FunEver/saved_models/08-26-15:13:35_simple_nn_doc/i(7000)_epoch(1)_(tra_score:0.9164416441644164|pr:0.4283778377837277|rec:0.8746624662466247|f1:0.575095052581864)"
    model_path = "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/i(9000)_epoch(1)_(tra_score:0.9212421242124212|pr:0.4299679967996279|rec:0.8818631863186318|f1:0.5780819247968391)"

    dev_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/dev.jsonl"
    # train_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/train.jsonl"
    dev_data_list = common.load_jsonl(dev_upstream_file)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    cursor = fever_db.get_cursor()
    complete_upstream_dev_data = disamb.sample_disamb_inference(common.load_jsonl(dev_upstream_file), cursor,
                                                                contain_first_sentence=contain_first_sentence)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=dev_batch_size)

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

    model.load_state_dict(torch.load(model_path))
    model.display()
    model.to(device)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)
    common.save_jsonl(complete_upstream_dev_data,
                      "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/ablation_neural_doc.jsonl")

    disamb.enforce_disabuigation_into_retrieval_result_v1(complete_upstream_dev_data,
                                                          dev_data_list)
    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    print("Strict score:", oracle_score)
    print(f"Eval Tracking score:", f"{oracle_score}")


def pipeline_function(upstream_file, model_path):
    seed = 12
    batch_size = 128
    dev_batch_size = 128
    lazy = True
    torch.manual_seed(seed)
    contain_first_sentence = True

    dev_upstream_file = upstream_file
    dev_data_list = common.load_jsonl(dev_upstream_file)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    cursor = fever_db.get_cursor()
    complete_upstream_dev_data = disamb.sample_disamb_inference(common.load_jsonl(dev_upstream_file), cursor,
                                                                contain_first_sentence=contain_first_sentence)
    print("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=dev_batch_size)

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

    model.load_state_dict(torch.load(model_path))
    model.display()
    model.to(device)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)
    # common.save_jsonl(complete_upstream_dev_data,
    #                   "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/ablation_neural_doc.jsonl")

    # remember to add this back
    # disamb.enforce_disabuigation_into_retrieval_result_v1(complete_upstream_dev_data,
    #                                                       dev_data_list)

    dev_doc_score_list = complete_upstream_dev_data
    return dev_doc_score_list
    # oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

    # print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    # print("Strict score:", oracle_score)
    # print(f"Eval Tracking score:", f"{oracle_score}")


def utest_results_debug():
    cursor = fever_db.get_cursor()
    contain_first_sentence = True
    dev_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/dev.jsonl"
    # dev_upstream_file = config.RESULT_PATH / "doc_retri/docretri.pageview/dev.jsonl"
    # train_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/train.jsonl"
    dev_data_list = common.load_jsonl(dev_upstream_file)
    # disamb.item_remove_old_rule(dev_data_list)
    disamb.item_resorting(dev_data_list, 5)

    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)
    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")

    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=10)
    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")

    # Dev(raw_acc/pr/rec/f1):0.9198419841984199/0.4589658965896083/0.8797629762976298/0.6032307383226128    length: 3.25
    # Dev(raw_acc/pr/rec/f1):0.9202920292029203/0.5114148914891038/0.8804380438043804/0.6470067565581602    length: 2.89
    #
    # exit(0)
    # complete_upstream_dev_data = disamb.sample_disamb_inference(common.load_jsonl(dev_upstream_file), cursor,
    #                                                             contain_first_sentence=contain_first_sentence)

    complete_upstream_dev_data = common.load_jsonl(
        "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/ablation_neural_doc.jsonl")
    disamb.enforce_disabuigation_into_retrieval_result_v2(complete_upstream_dev_data,
                                                          dev_data_list, prob_sh=0.00005)

    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    print("Strict score:", oracle_score)
    print(f"Eval Tracking score:", f"{oracle_score}")

    disamb.item_resorting(dev_data_list, 10)

    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=10)

    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    print("Strict score:", oracle_score)
    print(f"Eval Tracking score:", f"{oracle_score}")

    # Eval Tracking score: 0.9202420242024203 prob


# Eval Tracking score: 0.9202420242024203 score


def utest_results():
    cursor = fever_db.get_cursor()
    contain_first_sentence = True
    dev_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/dev.jsonl"
    #
    # dev_upstream_file = config.RESULT_PATH / "doc_retri/docretri.pageview/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/train.jsonl"
    dev_data_list = common.load_jsonl(dev_upstream_file)
    # disamb.item_resorting(dev_data_list)
    # disamb.item_remove_old_rule(dev_data_list)
    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)
    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    print("Strict score:", oracle_score)
    print(f"Eval Tracking score:", f"{oracle_score}")

    # oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=10)
    #
    # print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    # print("Strict score:", oracle_score)
    # print(f"Eval Tracking score:", f"{oracle_score}")

    # Dev(raw_acc/pr/rec/f1):0.9198419841984199/0.4589658965896083/0.8797629762976298/0.6032307383226128    length: 3.25
    # Dev(raw_acc/pr/rec/f1):0.9202920292029203/0.5114148914891038/0.8804380438043804/0.6470067565581602    length: 2.89

    # print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    #
    # exit(0)
    # complete_upstream_dev_data = disamb.sample_disamb_inference(common.load_jsonl(dev_upstream_file), cursor,
    #                                                             contain_first_sentence=contain_first_sentence)

    complete_upstream_dev_data = common.load_jsonl(
        "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/ablation_neural_doc.jsonl")
    disamb.enforce_disabuigation_into_retrieval_result_v2(complete_upstream_dev_data,
                                                          dev_data_list, prob_sh=0.0001)

    oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)
    print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    print("Strict score:", oracle_score)
    print(f"Eval Tracking score:", f"{oracle_score}")

    # oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=10)
    #
    # print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    # print("Strict score:", oracle_score)
    # print(f"Eval Tracking score:", f"{oracle_score}")

    # Eval Tracking score: 0.9202420242024203 prob
    # Eval Tracking score: 0.9202420242024203 score


def build_relatedness_for_train():
    seed = 12
    batch_size = 128
    dev_batch_size = 128
    lazy = True
    torch.manual_seed(seed)
    contain_first_sentence = True
    pn_ratio = 1.0
    # keep_neg_sample_prob = 0.4
    # sample_prob_decay = 0.05

    # model_path = "/home/easonnie/projects/FunEver/saved_models/08-26-15:13:35_simple_nn_doc/i(7000)_epoch(1)_(tra_score:0.9164416441644164|pr:0.4283778377837277|rec:0.8746624662466247|f1:0.575095052581864)"
    model_path = "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/i(9000)_epoch(1)_(tra_score:0.9212421242124212|pr:0.4299679967996279|rec:0.8818631863186318|f1:0.5780819247968391)"

    dev_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/dev.jsonl"
    # train
    # train_upstream_file = config.RESULT_PATH / "doc_retri_bls/docretri.basic.nopageview/train.jsonl"
    # dev_data_list = common.load_jsonl(dev_upstream_file)
    train_upstream_file = dev_upstream_file
    train_data_list = common.load_jsonl(train_upstream_file)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    cursor = fever_db.get_cursor()
    # complete_upstream_dev_data = disamb.sample_disamb_inference(common.load_jsonl(dev_upstream_file), cursor,
    #                                                             contain_first_sentence=contain_first_sentence)
    #
    complete_upstream_train_data = disamb.sample_disamb_training_v0(
        train_data_list,
        cursor, pn_ratio, contain_first_sentence,
        only_found=False)
    # complete_upstream_train_data = complete_upstream_dev_data

    print("Train size:", len(complete_upstream_train_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_train_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=dev_batch_size)

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

    model.load_state_dict(torch.load(model_path))
    model.display()
    model.to(device)

    eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_train_data)
    common.save_jsonl(complete_upstream_dev_data,
                      "/home/easonnie/projects/FunEver/saved_models/08-26-15:46:10_simple_nn_doc_first_sent/extra_needed_training_data/dev_doc_list.jsonl")

    # disamb.enforce_disabuigation_into_retrieval_result_v1(complete_upstream_dev_data,
    #                                                       dev_data_list)
    # oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)
    #
    # print(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
    # print("Strict score:", oracle_score)
    # print(f"Eval Tracking score:", f"{oracle_score}")


def debug_fever():
    num_epoch = 8
    seed = 12
    batch_size = 128
    experiment_name = "simple_nn"
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 0.6
    sample_prob_decay = 0.1

    dev_upstream_file = config.RESULT_PATH / "doc_retri/cn_util_Jul17_docretri.singularize/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "doc_retri/cn_util_Jul17_docretri.singularize/train.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=300)
    # dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=False)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=300)

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
                  embedding_dim=300, max_l=280, num_of_class=2)

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
    i_epoch = 0

    start_lr = 0.0002
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

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


def pipeline_first_sent_selection(org_t_file, upstream_in_file, model_save_path):
    batch_size = 128
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

    complete_upstream_dev_data = get_full_list(org_t_file, dev_upstream_file, pred=True)
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


def get_score_multihop(t_data_file, additional_file, model_path, item_key='prioritized_docids_aside', top_k=6):
    batch_size = 64
    lazy = True

    SAVE_PATH = model_path
    print("Model From:", SAVE_PATH)

    additional_sentence_list = get_additional_list(t_data_file, additional_file, item_key=item_key, top_k=top_k)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy)

    print("Additional Dev size:", len(additional_sentence_list))
    dev_instances = dev_fever_data_reader.read(additional_sentence_list)

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
    additional_sentence_list = hidden_eval(model, eval_iter, additional_sentence_list)

    return additional_sentence_list


if __name__ == "__main__":
    # train_fever_v1()
    # doc_model_eval()
    # utest_results()
    utest_results_debug()
    # build_relatedness_for_train()