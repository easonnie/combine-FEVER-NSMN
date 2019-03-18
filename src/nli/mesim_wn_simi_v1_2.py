# This is the one with shortcut
import torch
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules import Embedding, Elmo

from data_util.data_readers.fever_reader_with_wn_simi import WNSIMIReader
from neural_modules.ema import EMA
from sample_for_nli.adv_sampler_v01 import get_adv_sampled_data
from sentence_retrieval.nn_postprocess_ablation import score_converter_scaled
from torch import nn
import copy
from neural_modules.ema import load_ema_to_model, save_ema_to_file
import os

import config

from data_util.exvocab import load_vocab_embeddings
from simi_sampler_nli_v0.simi_sampler import paired_selection_score_dict, threshold_sampler, \
    select_sent_with_prob_for_eval, adv_simi_sample_with_prob_v1_0, adv_simi_sample_with_prob_v1_1, \
    select_sent_with_prob_for_eval_list
from utils import common

from log_util import save_tool

from flint import torch_util
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from neural_modules import biDafAttn
from utils import c_scorer
from wn_featurizer import wn_persistent_api


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

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
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

    def raw_input_to_esim_input(self, s_tokens, s_elmo_chars, s_wn_feature):
        s_tokens = torch_util.length_truncate(s_tokens, self.max_l)
        s1_glove_embd = self.glove_embd_layer(s_tokens)
        s1_elmo_out = self.elmo_embd_layer(s_elmo_chars)
        s1_wn_feature = torch_util.length_truncate(s_wn_feature, self.max_l)
        s1_elmo_embd = torch_util.length_truncate(s1_elmo_out, self.max_l, is_elmo=True)

        s1_mask, s1_len = torch_util.get_length_and_mask(s_tokens)
        assert torch.equal(s1_elmo_embd['mask'], s1_mask)

        return s1_glove_embd, s1_elmo_embd['elmo_representations'][0], s1_wn_feature, s1_len

    def forward(self, batch):
        s1_tokens = batch['premise']['tokens'].to(next(self.parameters()).device)
        s1_elmo_chars = batch['premise']['elmo_chars'].to(next(self.parameters()).device)
        s1_wn_f_vector = batch['p_wn_feature'].to(next(self.parameters()).device)

        s2_tokens = batch['hypothesis']['tokens'].to(next(self.parameters()).device)
        s2_elmo_chars = batch['hypothesis']['elmo_chars'].to(next(self.parameters()).device)
        s2_wn_f_vector = batch['h_wn_feature'].to(next(self.parameters()).device)

        s1_glove_embd, s1_elmo_embd, s1_wn_f_vector, s1_len = self.raw_input_to_esim_input(
            s1_tokens, s1_elmo_chars, s1_wn_f_vector)
        s2_glove_embd, s2_elmo_embd, s2_wn_f_vector, s2_len = self.raw_input_to_esim_input(
            s2_tokens, s2_elmo_chars, s2_wn_f_vector)

        # Important difference!
        s1_layer1_in = torch.cat((s1_glove_embd, s1_elmo_embd, s1_wn_f_vector), dim=2)
        s1_layer2_in = torch.cat((s1_elmo_embd, s1_wn_f_vector), dim=2)

        s2_layer1_in = torch.cat((s2_glove_embd, s2_elmo_embd, s2_wn_f_vector), dim=2)
        s2_layer2_in = torch.cat((s2_elmo_embd, s2_wn_f_vector), dim=2)

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
    # SUPPORTS < (-.-) > 0
    # REFUTES < (-.-) > 1
    # NOT ENOUGH INFO < (-.-) > 2

    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    print("Evaluating ...")
    model.eval()
    n_correct = loss = 0
    totoal_size = 0

    y_pred_list = []
    y_true_list = []
    y_id_list = []

    with torch.no_grad():  # Important fixing.

        for batch_idx, batch in enumerate(data_iter):
            out = model(batch)
            y = batch['label']
            y_id_list.extend(list(batch['pid']))

            n_correct += (torch.max(out, 1)[1].view(y.size()) == y).sum().item()

            y_pred_list.extend(torch.max(out, 1)[1].view(y.size()).tolist())
            y_true_list.extend(y.tolist())

            loss += criterion(out, y).item() * y.size(0)
            totoal_size += y.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_list) == len(dev_data_list)
        assert len(y_true_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['id'])
            # Matching id

            dev_data_list[i]['predicted_label'] = id2label[y_pred_list[i]]
            # Reset neural set
            if len(dev_data_list[i]['predicted_sentids']) == 0:
                dev_data_list[i]['predicted_label'] = "NOT ENOUGH INFO"

                # dev_data_list[i]['predicted_evidence'] = convert_evidence2scoring_format(dev_data_list[i]['predicted_sentids'])

        print('n_correct:', n_correct)
        print('total_size:', totoal_size)

        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_data_list, dev_data_list, mode=eval_mode,
                                                                    verbose=False)
        print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

        avg_acc = 100. * n_correct / totoal_size
        avg_loss = loss / totoal_size

    return strict_score, avg_loss


def eval_fever():
    # save_path = "/home/easonnie/projects/MiscEnc/saved_models/06-07-21:58:06_esim_elmo/i(60900)_epoch(4)_um_dev(80.03458096013019)_m_dev(79.174732552216)_seed(12)"
    save_path = "/home/easonnie/projects/MiscEnc/saved_models/07-02-14:40:01_esim_elmo_linear_amr_cs_score_filtering_0.5/i(5900)_epoch(3)_um_dev(39.73759153783564)_m_dev(40.18339276617422)_seed(12)"
    # save_path = "/home/easonnie/projects/MiscEnc/saved_models/07-02-14:42:34_esim_elmo_cs_score_filtering_0.7/i(1300)_epoch(4)_um_dev(32.55695687550855)_m_dev(32.42995415180846)_seed(12)"
    batch_size = 32

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    csnli_dataset_reader = CNLIReader(token_indexers=token_indexers,
                                      example_filter=lambda x: float(x['cs_score']) >= 0.7)

    # mnli_train_data_path = config.DATA_ROOT / "mnli/multinli_1.0_train.jsonl"
    mnli_m_dev_data_path = config.DATA_ROOT / "amrs/mnli_amr_ln/mnli_mdev.jsonl.cs"
    mnli_um_dev_data_path = config.DATA_ROOT / "amrs/mnli_amr_ln/mnli_umdev.jsonl.cs"

    # mnli_train_instances = csnli_dataset_reader.read(mnli_train_data_path)
    mnli_m_dev_instances = csnli_dataset_reader.read(mnli_m_dev_data_path)
    mnli_um_dev_instances = csnli_dataset_reader.read(mnli_um_dev_data_path)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300)

    model.load_state_dict(torch.load(save_path))

    model.display()
    model.to(device)

    # Create Log File

    criterion = nn.CrossEntropyLoss()

    eval_iter = biterator(mnli_m_dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    m_dev_score, m_dev_loss = eval_model(model, eval_iter, criterion)

    eval_iter = biterator(mnli_um_dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    um_dev_score, um_dev_loss = eval_model(model, eval_iter, criterion)

    print(f"Dev(M):{m_dev_score}/{m_dev_loss}")
    print(f"Dev(UM):{um_dev_score}/{um_dev_loss}")


def get_sampled_data(tokenized_data_file, additional_data_file):
    # This is for sampling training data.
    sampled_d_list = sample_v1_0(tokenized_data_file, additional_data_file, tokenized=True)
    return sampled_d_list


def get_actual_data(tokenized_data_file, additional_data_file):
    # This is for get actual data.
    actual_d_list = select_sent_for_eval(tokenized_data_file, additional_data_file, tokenized=True)
    return actual_d_list


def hidden_eval(model, data_iter, dev_data_list, with_logits=False, with_probs=False):
    # SUPPORTS < (-.-) > 0
    # REFUTES < (-.-) > 1
    # NOT ENOUGH INFO < (-.-) > 2

    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }

    print("Evaluating ...")
    with torch.no_grad():
        model.eval()
        totoal_size = 0

        y_pred_list = []
        y_id_list = []
        y_logits_list = []
        y_probs_list = []

        # if append_text:
        # y_premise = []
        # y_hypothesis = []

        for batch_idx, batch in enumerate(tqdm(data_iter)):
            out = model(batch)
            y_id_list.extend(list(batch['pid']))

            # if append_text:
            # y_premise.extend(list(batch['text']))
            # y_hypothesis.extend(list(batch['query']))

            y_pred_list.extend(torch.max(out, 1)[1].view(out.size(0)).tolist())

            if with_logits:
                y_logits_list.extend(out.tolist())

            if with_probs:
                y_probs_list.extend(F.softmax(out, dim=1).tolist())

            totoal_size += out.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_pred_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['id'])

            # Matching id
            dev_data_list[i]['predicted_label'] = id2label[y_pred_list[i]]
            if with_logits:
                dev_data_list[i]['logits'] = y_logits_list[i]

            if with_probs:
                dev_data_list[i]['probs'] = y_probs_list[i]

            # Reset neural set
            if len(dev_data_list[i]['predicted_sentids']) == 0:
                dev_data_list[i]['predicted_label'] = "NOT ENOUGH INFO"

            # if append_text:
            #     dev_data_list[i]['premise'] = y_premise[i]
            #     dev_data_list[i]['hypothesis'] = y_hypothesis[i]

        print('total_size:', totoal_size)

    return dev_data_list


def hidden_eval_fever():
    batch_size = 64
    lazy = True

    SAVE_PATH = "/home/easonnie/projects/FunEver/saved_models/07-18-21:07:28_m_esim_wn_elmo_sample_fixed/i(57000)_epoch(8)_dev(0.5755075507550755)_loss(1.7175163737963839)_seed(12)"

    dev_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    dev_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)

    complete_upstream_dev_data = get_actual_data(config.T_FEVER_DEV_JSONL, dev_upstream_file)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)
    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    # dev_biterator = BasicIterator(batch_size=batch_size * 2)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 300),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.display()
    model.to(device)

    eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    builded_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

    eval_mode = {'check_sent_id_correct': True, 'standard': True}

    for item in builded_dev_data:
        del item['label']

    print(c_scorer.fever_score(builded_dev_data, common.load_jsonl(config.T_FEVER_DEV_JSONL), mode=eval_mode))


def hidden_eval_fever_adv_v1():
    batch_size = 64
    lazy = True
    dev_prob_threshold = 0.5

    SAVE_PATH = "/home/easonnie/projects/FunEver/saved_models/07-20-22:28:24_mesim_wn_450_adv_sample_v1_|t_prob:0.35|top_k:8/i(46000)_epoch(7)_dev(0.6405140514051405)_loss(1.0761665150348825)_seed(12)"

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    upstream_dev_list = score_converter_scaled(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                               scale_prob=dev_prob_threshold,
                                               delete_prob=False)

    dev_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)

    complete_upstream_dev_data = get_actual_data(config.T_FEVER_DEV_JSONL, upstream_dev_list)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.display()
    model.to(device)

    eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    builded_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

    eval_mode = {'check_sent_id_correct': True, 'standard': True}

    common.save_jsonl(builded_dev_data, config.RESULT_PATH / "nli_results" / "pipeline_results_1.jsonl")
    c_scorer.delete_label(builded_dev_data)
    print(c_scorer.fever_score(builded_dev_data, common.load_jsonl(config.FEVER_DEV_JSONL), mode=eval_mode))


def train_fever():
    num_epoch = 10
    seed = 12
    batch_size = 32
    experiment_name = "m_esim_wn_elmo_sample_fixed"
    lazy = True

    dev_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/train.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    train_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)
    dev_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)

    complete_upstream_dev_data = get_actual_data(config.T_FEVER_DEV_JSONL, dev_upstream_file)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)
    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=batch_size * 2)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 300),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
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
        complete_upstream_train_data = get_sampled_data(config.T_FEVER_TRAIN_JSONL, train_upstream_file)

        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 6:
                mod = 5000
                # mod = 5
            else:
                mod = 500

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)

        # Save some cache wordnet feature.
        wn_persistent_api.persistence_update(p_dict)


def train_fever_v1_advsample():
    num_epoch = 12
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.35
    train_sample_top_k = 12
    experiment_name = f"mesim_wn_simi_v12|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
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
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_0(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 6:
                # mod = 5000
                mod = 10000
            else:
                mod = 500

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)

        # Save some cache wordnet feature.
        wn_persistent_api.persistence_update(p_dict)


def train_fever_v1_advsample_v2_shuffle_bigger():
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.5
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_nei_v2_non_shuffle_bigger|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320,
                                           shuffle_sentences=False)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 600 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(600, 600),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
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
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 8:
                # mod = 5000
                mod = 10000
            else:
                mod = 1000

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)


def train_fever_v1_advsample_v2_shuffle_bigger_ema():
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.5
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_nei_v2_non_shuffle_bigger_ema|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"
    resume_model = str(
        config.PRO_ROOT / "saved_models/saved_v1_2/i(77000)_epoch(11)_dev(0.6552155215521552)_loss(1.5343008287871691)_seed(12)")

    print("Do EMA:")

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320,
                                           shuffle_sentences=False)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 600 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(600, 600),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
    if resume_model is not None:
        print("Resume From:", resume_model)
        model.load_state_dict(torch.load(resume_model))
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

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # Resampling
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            # EMA update
            ema(model.named_parameters())

            mod = 500

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev:{strict_score}/{acc_score}")

                save_path = os.path.join(
                    file_path_prefix,
                    f'i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                )

                torch.save(model.state_dict(), save_path)

                # EMA saving
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                load_ema_to_model(cloned_empty_model, ema)
                complete_upstream_dev_data = hidden_eval(cloned_empty_model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score EMA(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev EMA:{strict_score}/{acc_score}")

                ema_save_path = os.path.join(
                    file_path_prefix,
                    f'ema_i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                )

                save_ema_to_file(ema, ema_save_path)
                # torch.save(model.state_dict(), save_path)


def train_fever_v1_advsample_v2_shuffle_std_ema(resume_model=None):
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.5
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_nei_v2_non_shuffle_std_ema_lr1|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"
    # resume_model = None

    print("Do EMA:")

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320,
                                           shuffle_sentences=False)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
    if resume_model is not None:
        model.load_state_dict(torch.load(resume_model))
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

    start_lr = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # Resampling
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            # EMA update
            ema(model.named_parameters())

            mod = 200

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev:{strict_score}/{acc_score}")

                save_path = os.path.join(
                    file_path_prefix,
                    f'i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                )

                torch.save(model.state_dict(), save_path)

                # EMA saving
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                load_ema_to_model(cloned_empty_model, ema)
                complete_upstream_dev_data = hidden_eval(cloned_empty_model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score EMA(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev EMA:{strict_score}/{acc_score}")

                ema_save_path = os.path.join(
                    file_path_prefix,
                    f'ema_i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                )

                save_ema_to_file(ema, ema_save_path)
                # torch.save(model.state_dict(), save_path)


def train_fever_v1_advsample_v2_shuffle_std_ema_on_dev(resume_model=None):
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.5
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_nei_v2_non_shuffle_std_ema_lr1_on_dev|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"
    # resume_model = None

    print("Do EMA:")

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_l = common.load_jsonl(config.RESULT_PATH /
                                              "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    dev_train_upstream_sent_list = dev_upstream_sent_list + train_upstream_sent_l

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    remaining_dev_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_dev_sent.jsonl")

    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_dev_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320,
                                           shuffle_sentences=False)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
    if resume_model is not None:
        model.load_state_dict(torch.load(resume_model))
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

    start_lr = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # Resampling
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_DT_JSONL, dev_train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_DT_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            # EMA update
            ema(model.named_parameters())

            mod = 5000

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev:{strict_score}/{acc_score}")

                save_path = os.path.join(
                    file_path_prefix,
                    f'i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                )

                torch.save(model.state_dict(), save_path)

                # EMA saving
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                load_ema_to_model(cloned_empty_model, ema)
                complete_upstream_dev_data = hidden_eval(cloned_empty_model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score EMA(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev EMA:{strict_score}/{acc_score}")

                ema_save_path = os.path.join(
                    file_path_prefix,
                    f'ema_i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                )

                save_ema_to_file(ema, ema_save_path)
                # torch.save(model.state_dict(), save_path)


def train_fever_v1_advsample_v2_shuffle_bigger_train_on_dev():
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.5
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_nei_v2_non_shuffle_bigger_on_dev|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_l = common.load_jsonl(config.RESULT_PATH /
                                              "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    dev_train_upstream_sent_list = dev_upstream_sent_list + train_upstream_sent_l

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    remaining_dev_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_dev_sent.jsonl")

    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_dev_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320,
                                           shuffle_sentences=False)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 600 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(600, 600),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
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
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_DT_JSONL, dev_train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_DT_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 8:
                # mod = 5000
                mod = 10000
            else:
                mod = 1000

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)


def train_fever_v1_advsample_shuffle_bigger():
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.35
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_shuffle_bigger|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict,
                                           max_l=320,
                                           shuffle_sentences=True)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 600 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(600, 600),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
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
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_0(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch < 10:
                # mod = 5000
                mod = 10000
            else:
                mod = 500

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)


# Save some cache wordnet feature.
# Don't save to avoid IO problem
# wn_persistent_api.persistence_update(p_dict)


def train_fever_v1_advsample_shuffle_stad():
    num_epoch = 12 + 2
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.35
    train_sample_top_k = 10
    experiment_name = f"mesim_wn_simi_v12_shuffle_stad|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Mac OS
    # dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                            "sent_retri_nn/2018_07_20_15-17-59_r/dev_sent.jsonl")
    #
    # train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
    #                                              "sent_retri_nn/2018_07_20_15-17-59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=320,
                                           shuffle_sentences=True)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
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
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_0(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch < 10:
                # mod = 5000
                mod = 10000
            else:
                mod = 500

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                print(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)

        # Save some cache wordnet feature.
        # Don't save to avoid IO problem
        # wn_persistent_api.persistence_update(p_dict)


def utest_data_loader():
    num_epoch = 8
    seed = 12
    batch_size = 32
    experiment_name = "mesim_wn_elmo"
    lazy = True

    dev_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "sent_retri/2018_07_05_17:17:50_r/train.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    train_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict)
    dev_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict)

    complete_upstream_dev_data = get_actual_data(config.T_FEVER_DEV_JSONL, dev_upstream_file)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)
    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    # Build Model
    complete_upstream_train_data = get_sampled_data(config.T_FEVER_TRAIN_JSONL, train_upstream_file)[:20000]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0
    sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)
    train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)

    for i, batch in tqdm(enumerate(train_iter)):
        pass
        batch['p_wn_feature']
        batch['h_wn_feature']
        # print(batch.keys())
        # print(batch['p_wn_feature'])
        # print(batch['h_wn_feature'])
    wn_persistent_api.persistence_update(p_dict)


def spectrum_eval_manual_check():
    batch_size = 64
    lazy = True

    SAVE_PATH = "/home/easonnie/projects/FunEver/saved_models/07-18-21:07:28_m_esim_wn_elmo_sample_fixed/i(57000)_epoch(8)_dev(0.5755075507550755)_loss(1.7175163737963839)_seed(12)"

    # IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_17_15:52:19_r/dev_sent.jsonl"
    IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl"
    # IN_FILE = config.RESULT_PATH / "sent_retri_nn/2018_07_17_16-34-19_r/dev_sent.jsonl"
    dev_sent_result_lsit = common.load_jsonl(IN_FILE)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    p_dict = wn_persistent_api.persistence_load()
    dev_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)
    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 300),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.display()
    model.to(device)

    for sc_prob in [0.2, 0.3, 0.4]:
        upstream_dev_list = score_converter_scaled(config.T_FEVER_DEV_JSONL, dev_sent_result_lsit, scale_prob=sc_prob,
                                                   delete_prob=False)

        dev_fever_data_reader = WNReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)
        complete_upstream_dev_data = get_actual_data(config.T_FEVER_DEV_JSONL, upstream_dev_list)
        dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

        eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
        builded_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

        print("------------------------------------")
        print("Scaling_prob:", sc_prob)
        eval_mode = {'check_sent_id_correct': True, 'standard': True}

        for item in builded_dev_data:
            del item['label']

        print(c_scorer.fever_score(builded_dev_data, common.load_jsonl(config.T_FEVER_DEV_JSONL), mode=eval_mode))
        # del upstream_dev_list
        # del complete_upstream_dev_data
        del dev_fever_data_reader
        del dev_instances
        print("------------------------------------")


def pipeline_nli_run(t_org_file, upstream_dev_data_list, upstream_sent_file_list, model_path,
                     with_logits=False, with_probs=False, load_from_dict=False):
    batch_size = 16
    lazy = True

    print("Size:", len(upstream_dev_data_list))

    print("Building Prob Dicts...")
    selection_dict = dict()
    for upstream_sent_file in upstream_sent_file_list:
        upstream_sent_l = common.load_jsonl(upstream_sent_file)
        selection_dict = paired_selection_score_dict(upstream_sent_l, selection_dict)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=420)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(t_org_file, upstream_dev_data_list,
                                                                selection_dict, tokenized=True, pipeline=True)
    # This the key function that use convert upstream_sent_list and original_file to NLI model consumerable file.

    complete_upstream_dev_data = append_hidden_label(complete_upstream_dev_data)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=400)

    print("Model Max length:", model.max_l)
    if not load_from_dict:
        model.load_state_dict(torch.load(model_path))
    else:
        load_ema_to_model(model, model_path)

    model.display()
    model.to(device)

    eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data,
                                             with_logits=with_logits,
                                             with_probs=with_probs)

    wn_persistent_api.persistence_update(p_dict)

    return complete_upstream_dev_data


def pipeline_nli_run_list(t_org_file, upstream_dev_data_list, upstream_sent_file_list, model_path,
                          with_logits=False, with_probs=False, load_from_dict=False):
    batch_size = 16
    lazy = True

    print("Size:", len(upstream_dev_data_list))

    print("Building Prob Dicts...")
    selection_dict = dict()
    for upstream_sent_file in upstream_sent_file_list:
        # upstream_sent_l = common.load_jsonl(upstream_sent_file)
        selection_dict = paired_selection_score_dict(upstream_sent_file, selection_dict)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=420)

    complete_upstream_dev_data = select_sent_with_prob_for_eval_list(t_org_file, upstream_dev_data_list,
                                                                     selection_dict, tokenized=True, pipeline=True)

    complete_upstream_dev_data = append_hidden_label(complete_upstream_dev_data)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=400)

    print("Model Max length:", model.max_l)
    if not load_from_dict:
        model.load_state_dict(torch.load(model_path))
    else:
        load_ema_to_model(model, model_path)

    model.display()
    model.to(device)

    eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data,
                                             with_logits=with_logits,
                                             with_probs=with_probs)

    # wn_persistent_api.persistence_update(p_dict)

    return complete_upstream_dev_data


class ModelFORDemo:
    def __init__(self, model_path):
        # Prepare Data
        lazy = False
        token_indexers = {
            'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
            'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
        }

        p_dict = wn_persistent_api.persistence_load()

        dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=420)

        vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
        vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

        # Build Model
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
        # device_num = -1 if device.type == 'cpu' else 0

        device = torch.device("cpu")
        device_num = -1 if device.type == 'cpu' else 0

        biterator = BasicIterator(batch_size=16)
        biterator.index_with(vocab)

        model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                                   1024 + 450 + dev_fever_data_reader.wn_feature_size),
                      rnn_size_out=(450, 450),
                      weight=weight_dict['glove.840B.300d'],
                      vocab_size=vocab.get_vocab_size('tokens'),
                      mlp_d=900,
                      embedding_dim=300, max_l=400)

        model.display()
        model.to(device)
        model.load_state_dict(torch.load(model_path))

        self.model = model
        self.dev_fever_data_reader = dev_fever_data_reader
        self.device_num = device_num
        self.biterator = biterator

    def evaluate(self, t_org_file, upstream_dev_data_list, upstream_sent_file_list,
                 with_logits=False, with_probs=False):
        print("Building Prob Dicts...")
        selection_dict = dict()
        for upstream_sent_file in upstream_sent_file_list:
            # upstream_sent_l = common.load_jsonl(upstream_sent_file)
            selection_dict = paired_selection_score_dict(upstream_sent_file, selection_dict)

        # Prepare Data
        print("Prepare Data")
        complete_upstream_dev_data = select_sent_with_prob_for_eval_list(t_org_file, upstream_dev_data_list,
                                                                         selection_dict, tokenized=True, pipeline=True)

        # complete_upstream_dev_data = append_hidden_label(complete_upstream_dev_data)
        print("Read Data")
        dev_instances = self.dev_fever_data_reader.read(complete_upstream_dev_data)

        # Load Vocabulary
        # biterator = BasicIterator(batch_size=16)

        model = self.model
        print("Indexing")
        eval_iter = self.biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=self.device_num)
        complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data,
                                                 with_logits=with_logits,
                                                 with_probs=with_probs)

        return complete_upstream_dev_data


def pipeline_nli_run_bigger(t_org_file, upstream_dev_data_list, upstream_sent_file_list, model_path,
                            with_logits=False, with_probs=False):
    batch_size = 16
    lazy = True

    print("Size:", len(upstream_dev_data_list))

    print("Building Prob Dicts...")
    selection_dict = dict()
    for upstream_sent_file in upstream_sent_file_list:
        upstream_sent_l = common.load_jsonl(upstream_sent_file)
        selection_dict = paired_selection_score_dict(upstream_sent_l, selection_dict)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=450)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(t_org_file, upstream_dev_data_list,
                                                                selection_dict, tokenized=True, pipeline=True)
    complete_upstream_dev_data = append_hidden_label(complete_upstream_dev_data)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 600 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(600, 600),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=400)

    print("Model Max length:", model.max_l)
    model.load_state_dict(torch.load(model_path))
    model.display()
    model.to(device)

    eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
    complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data,
                                             with_logits=with_logits,
                                             with_probs=with_probs)

    wn_persistent_api.persistence_update(p_dict)

    return complete_upstream_dev_data


def append_hidden_label(d_list):
    for item in d_list:
        item['label'] = 'hidden'
    return d_list


if __name__ == "__main__":
    # train_fever()
    train_fever_v1_advsample()
    # train_fever_v1_advsample_v2_shuffle_bigger_ema()
    # r_model = str(config.PRO_ROOT / "saved_models/07-22-14:41:21_mesim_wn_simi_v12_|t_prob:0.35|top_k:10/i(77000)_epoch(11)_dev(0.6601160116011601)_loss(1.1138329989302813)_seed(12)")
    # r_model = str(config.PRO_ROOT / "saved_models/07-22-14:41:21_mesim_wn_simi_v12_|t_prob:0.35|top_k:10/i(46000)_epoch(7)_dev(0.6573657365736574)_loss(1.057304784975978)_seed(12)")
    # train_fever_v1_advsample_v2_shuffle_std_ema(resume_model=r_model)
    # train_fever_v1_advsample_v2_shuffle_std_ema_on_dev(resume_model=r_model)

    # train_fever_v1_advsample_v2_shuffle_bigger_train_on_dev()
    # train_fever_v1_advsample_shuffle_bigger()
    train_fever_v1_advsample_shuffle_stad()
    # hidden_eval_fever_adv_v1()
    # hidden_eval_fever()
    # utest_data_loader()
    # spectrum_eval_manual_check()
