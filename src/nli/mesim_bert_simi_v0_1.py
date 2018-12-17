# This file is created on 22 Nov 2018 11:38.
# The original goal of this file is for bert-as-feature with NSMN on FEVER for NAACL-2019.
# The code is modified from: /home/easonnie/projects/FunEver/src/nli/mesim_wn_simi_v1_2.py

import torch
import numpy as np
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules import Embedding, Elmo

from data_util.data_readers.fever_bert_reader import BertReader
from data_util.data_readers.fever_bert_reader_sselection import BertSSReader
from data_util.data_readers.fever_reader_with_wn_simi import WNSIMIReader
from neural_modules.bert_servant import BertServant
from neural_modules.ema import EMA
from sample_for_nli.adv_sampler_v01 import get_adv_sampled_data
from sentence_retrieval.nn_postprocess_ablation import score_converter_scaled
from torch import nn
import copy
from neural_modules.ema import load_ema_to_model, save_ema_to_file
import os
import random

import config

from data_util.exvocab import load_vocab_embeddings, ExVocabulary
from sentence_retrieval.sampler_for_nmodel import get_full_list
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
from neural_modules import activation as actf
from utils import c_scorer
from simi_sampler_nli_v0.simi_sampler import threshold_sampler_insure_unique
from allennlp.modules import ScalarMix


class ESIM(nn.Module):
    # This is Modified ESIM sequence matching model
    # lstm
    def __init__(self, rnn_size_in=(1024, 1024), rnn_size_out=(300, 300), max_l=300,
                 mlp_d=300, num_of_class=3, drop_r=0.5, activation_type='gelu'):

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
        elif activation_type == 'gelu':
            activation = actf.GELU()
        else:
            raise ValueError("Not a valid activation!")

        self.activation = activation

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

        p_s1_coattentioned = self.activation(self.projection(s1_coattentioned))
        p_s2_coattentioned = self.activation(self.projection(s2_coattentioned))

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
    def __init__(self,
                 bert_servant, bert_batch_size=1,
                 rnn_size_in=(1024, 1024 + 300),
                 rnn_size_out=(300, 300), max_l=300,
                 mlp_d=300,
                 num_of_class=3,
                 drop_r=0.5,
                 activation_type='gelu'):

        super(Model, self).__init__()
        self.bert_mix_scalar = ScalarMix(4)
        self.esim_layer = ESIM(rnn_size_in, rnn_size_out, max_l, mlp_d, num_of_class, drop_r, activation_type)
        self.bert_servant = bert_servant
        self.bert_batch_size = bert_batch_size

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

    def raw_input_to_esim_input(self, batch, task_indicator):
        # for verification task_indicator = [1]
        paired_out = self.bert_servant.run_paired_seq(self.bert_batch_size, batch, self.bert_mix_scalar)
        p_seq, p_l, h_seq, h_l = self.bert_servant.paired_seq_split(paired_out,
                                                                    batch['bert_premise_span'],
                                                                    batch['bert_hypothesis_span'])

        # adding additional features:
        batch_premise_features_torch_list = []
        p_l_list = []
        for np_probs in batch['premise_probs']:
            np_feature = np.concatenate([np_probs, np.asarray([task_indicator] * len(np_probs), dtype=np.float32)], axis=1)
            batch_premise_features_torch_list.append(torch.from_numpy(np_feature))
            p_l_list.append(len(np_feature))
        premise_features_torch = torch_util.pack_list_sequence(batch_premise_features_torch_list, p_l_list)

        batch_hypothesis_features_torch_list = []
        h_l_list = []
        for np_probs in batch['hypothesis_probs']:
            np_feature = np.concatenate([np_probs, np.asarray([task_indicator] * len(np_probs), dtype=np.float32)], axis=1)
            batch_hypothesis_features_torch_list.append(torch.from_numpy(np_feature))
            h_l_list.append(len(np_feature))
        hypothesis_features_torch = torch_util.pack_list_sequence(batch_hypothesis_features_torch_list, h_l_list)

        assert premise_features_torch.size(1) == p_seq.size(1)
        assert premise_features_torch.size(0) == p_seq.size(0)
        assert hypothesis_features_torch.size(1) == h_seq.size(1)
        assert hypothesis_features_torch.size(0) == h_seq.size(0)

        premise_features_torch = premise_features_torch.to(next(self.parameters()).device)
        hypothesis_features_torch = hypothesis_features_torch.to(next(self.parameters()).device)

        p_fed_seq = torch.cat([p_seq, premise_features_torch], dim=2)
        h_fed_seq = torch.cat([h_seq, hypothesis_features_torch], dim=2)

        return p_fed_seq, p_l, h_fed_seq, h_l

    def forward(self, batch):
        s1_in, s1_len, s2_in, s2_len = self.raw_input_to_esim_input(batch, [1])
        # print(s1_layer1_in.size())
        # print(s1_layer2_in.size())
        # print(s2_layer1_in.size())
        # print(s2_layer2_in.size())
        esim_out = self.esim_layer(s1_in, s1_in, s1_len,
                                   s2_in, s2_in, s2_len)

        return esim_out


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


def train_fever_ema_v1(resume_model=None):
    """
    This method is training script for bert+nsmn model
    :param resume_model:
    :return:
    """
    num_epoch = 200
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.02
    train_prob_threshold = 0.02
    train_sample_top_k = 8
    experiment_name = f"bert_nsmn_ema_lr1|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    bert_type_name = "bert-large-uncased"
    bert_servant = BertServant(bert_type_name=bert_type_name)

    # print("Do EMA:")
    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/balanced_sentence_selection_results/dev_sent_pred_scores.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/balanced_sentence_selection_results/train_sent_scores.jsonl")
    # Prepare Data
    # 22 Nov 2018 03:16
    # Remove this because everything can be handled by Bert Servant.

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/balanced_sentence_selection_results/train_sent_scores.jsonl")

    dev_sent_list = common.load_jsonl(config.RESULT_PATH /
                                      "sent_retri_nn/balanced_sentence_selection_results/dev_sent_pred_scores.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler_insure_unique(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                                        prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = BertReader(bert_servant, lazy=lazy, max_l=60)
    train_fever_data_reader = BertReader(bert_servant, lazy=lazy, max_l=60)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)

    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary, if we are using bert, we don't need anything here.
    biterator = BasicIterator(batch_size=batch_size)

    unk_token_num = {'tokens': 2600}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace('SUPPORTS', namespace='labels')
    vocab.add_token_to_namespace('REFUTES', namespace='labels')
    vocab.add_token_to_namespace('NOT ENOUGH INFO', namespace='labels')
    print(vocab)

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0
    bert_servant.bert_model.to(device)

    # Init model here
    model = Model(bert_servant, bert_batch_size=1,
                  rnn_size_in=(1024 + 2, 1024 + 2 + 300),     # probs + task indicator.
                  rnn_size_out=(300, 300), max_l=250,
                  mlp_d=300,
                  num_of_class=3,
                  drop_r=0.5,
                  activation_type='gelu')
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
    #
    start_lr = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # Resampling
        train_data_with_candidate_sample_list = \
            threshold_sampler_insure_unique(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list,
                                            train_prob_threshold,
                                            top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)
        random.shuffle(complete_upstream_train_data)
        print("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)

            y = batch['label'].to(next(model.parameters()).device)

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            # EMA update
            # ema(model.named_parameters())

            if i_epoch < 15:
                mod = 20000
                # mod = 500
            else:
                mod = 2000

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev:{strict_score}/{acc_score}")

                # EMA saving
                # eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                # load_ema_to_model(cloned_empty_model, ema)
                # complete_upstream_dev_data = hidden_eval(cloned_empty_model, eval_iter, complete_upstream_dev_data)
                #
                # eval_mode = {'check_sent_id_correct': True, 'standard': True}
                # strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(complete_upstream_dev_data,
                #                                                             common.load_jsonl(config.T_FEVER_DEV_JSONL),
                #                                                             mode=eval_mode,
                #                                                             verbose=False)
                # print("Fever Score EMA(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)
                #
                # print(f"Dev EMA:{strict_score}/{acc_score}")

                need_save = False
                if strict_score > best_dev:
                    best_dev = strict_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)

                    # ema_save_path = os.path.join(
                    #     file_path_prefix,
                    #     f'ema_i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                    # )
                    #
                    # save_ema_to_file(ema, ema_save_path)
                    # torch.save(model.state_dict(), save_path)


def hidden_eval_on_sselection(model, data_iter, dev_data_list):
    # This method is created on 25 Nov 2018 09:32 to use the claim verifier model to do scoring for sentence selection.
    # S: 0
    # R: 1
    # NEI: 2
    # true 3
    # false 4

    # hidden < (-.-) > -2

    with torch.no_grad():
        id2label = {
            0: "S",
            1: "R",
            2: "NEI",
            -2: "hidden"
        }

        print("Evaluating ...")
        model.eval()
        totoal_size = 0

        y_s_logits_list = []
        y_s_prob_list = []

        y_r_logits_list = []
        y_r_prob_list = []

        y_nei_logits_list = []
        y_nei_prob_list = []

        y_id_list = []

        for batch_idx, batch in enumerate(tqdm(data_iter)):
            out = model(batch)
            prob = F.softmax(out, dim=1)

            y = batch['label']
            y_id_list.extend(list(batch['pid']))

            y_s_logits_list.extend(out[:, 0].tolist())
            y_s_prob_list.extend(prob[:, 0].tolist())

            y_r_logits_list.extend(out[:, 1].tolist())
            y_r_prob_list.extend(prob[:, 1].tolist())

            y_nei_logits_list.extend(out[:, 2].tolist())
            y_nei_prob_list.extend(prob[:, 2].tolist())

            totoal_size += y.size(0)

        # assert len(y_id_list) == len(dev_data_list)
        # assert len(y_s_logits_list) == len(dev_data_list)
        # assert len(y_r_logits_list) == len(dev_data_list)
        # assert len(y_nei_logits_list) == len(dev_data_list)
        #
        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['selection_id'])
            assert 1 - (y_s_prob_list[i] + y_r_prob_list[i] + y_nei_prob_list[i]) <= 1e-3
            # Matching id

            dev_data_list[i]['score_s'] = y_s_logits_list[i]
            dev_data_list[i]['prob_s'] = y_s_prob_list[i]

            dev_data_list[i]['score_r'] = y_r_logits_list[i]
            dev_data_list[i]['prob_r'] = y_r_prob_list[i]

            dev_data_list[i]['score_nei'] = y_nei_logits_list[i]
            dev_data_list[i]['prob_nei'] = y_nei_prob_list[i]
            # Reset neural set

        print('total_size:', totoal_size)

    return dev_data_list


def eval_m_on_sselection(model_path):
    # This method is created on 25 Nov 2018 09:32 to use the claim verifier model to do scoring for sentence selection.
    batch_size = 32
    lazy = True
    top_k_doc = 5
    save_file_name = "/home/easonnie/projects/FunEver/results/sent_retri_nn/bert_verification_for_selection_probing_11_25_2018/dev_sent_scores.txt"

    dev_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl"
    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True,
                                               top_k=top_k_doc)

    debug = None

    bert_type_name = "bert-large-uncased"
    bert_servant = BertServant(bert_type_name=bert_type_name)
    # train_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/train_doc.jsonl"

    # train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    # dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=False)
    dev_fever_data_reader = BertSSReader(bert_servant, lazy=lazy, max_l=80)

    print("Dev size:", len(complete_upstream_dev_data))
    # dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)
    if debug is not None:
        complete_upstream_dev_data = complete_upstream_dev_data[:debug]

    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    unk_token_num = {'tokens': 2600}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace('SUPPORTS', namespace='labels')
    vocab.add_token_to_namespace('REFUTES', namespace='labels')
    vocab.add_token_to_namespace('NOT ENOUGH INFO', namespace='labels')

    vocab.add_token_to_namespace('true', namespace='labels')
    vocab.add_token_to_namespace('false', namespace='labels')
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    print(vocab)

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0
    bert_servant.bert_model.to(device)

    # Init model here
    model = Model(bert_servant, bert_batch_size=1,
                  rnn_size_in=(1024 + 2, 1024 + 2 + 300),  # probs + task indicator.
                  rnn_size_out=(300, 300), max_l=250,
                  mlp_d=300,
                  num_of_class=3,
                  drop_r=0.5,
                  activation_type='gelu')

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1)
    dev_scored_data = hidden_eval_on_sselection(model, eval_iter, complete_upstream_dev_data)

    common.save_jsonl(dev_scored_data, save_file_name)


def train_fever_ema_v1_runtest(resume_model=None):
    """
    This method is training script for bert+nsmn model
    :param resume_model:
    :return:
    """
    num_epoch = 200
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.02
    train_prob_threshold = 0.02
    train_sample_top_k = 8
    experiment_name = f"bert_nsmn_ema_lr1|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}"

    bert_type_name = "bert-large-uncased"
    bert_servant = BertServant(bert_type_name=bert_type_name)

    print("Do EMA:")
    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/balanced_sentence_selection_results/dev_sent_pred_scores.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/balanced_sentence_selection_results/train_sent_scores.jsonl")
    # Prepare Data
    # 22 Nov 2018 03:16
    # Remove this because everything can be handled by Bert Servant.

    print("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/balanced_sentence_selection_results/train_sent_scores.jsonl")

    dev_sent_list = common.load_jsonl(config.RESULT_PATH /
                                      "sent_retri_nn/balanced_sentence_selection_results/dev_sent_pred_scores.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler_insure_unique(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                                        prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = BertReader(bert_servant, lazy=lazy, max_l=60)
    train_fever_data_reader = BertReader(bert_servant, lazy=lazy, max_l=60)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)

    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary, if we are using bert, we don't need anything here.
    biterator = BasicIterator(batch_size=batch_size)

    unk_token_num = {'tokens': 2600}  # work around for initiating vocabulary.
    vocab = ExVocabulary(unk_token_num=unk_token_num)
    vocab.add_token_to_namespace('SUPPORTS', namespace='labels')
    vocab.add_token_to_namespace('REFUTES', namespace='labels')
    vocab.add_token_to_namespace('NOT ENOUGH INFO', namespace='labels')
    print(vocab)

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0
    bert_servant.bert_model.to(device)

    # Init model here
    model = Model(bert_servant, bert_batch_size=1,
                  rnn_size_in=(1024 + 2, 1024 + 2 + 300),     # probs + task indicator.
                  rnn_size_out=(300, 300), max_l=250,
                  mlp_d=300,
                  num_of_class=3,
                  drop_r=0.5,
                  activation_type='gelu')
    model.to(device)

    # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{experiment_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    # best_dev = -1
    # iteration = 0

    # start_lr = 0.0001
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    # criterion = nn.CrossEntropyLoss()

    for i_epoch in range(num_epoch):
        dev_iter = biterator(dev_instances, shuffle=False, num_epochs=1)
        for i, batch in enumerate(dev_iter):
            out = model(batch)



def append_hidden_label(d_list):
    for item in d_list:
        item['label'] = 'hidden'
    return d_list


if __name__ == "__main__":
    # train_fever()
    # train_fever_v1_advsample()
    # train_fever_v1_advsample_v2_shuffle_bigger_ema()
    # r_model = str(config.PRO_ROOT / "saved_models/07-22-14:41:21_mesim_wn_simi_v12_|t_prob:0.35|top_k:10/i(77000)_epoch(11)_dev(0.6601160116011601)_loss(1.1138329989302813)_seed(12)")
    # r_model = str(config.PRO_ROOT / "saved_models/07-22-14:41:21_mesim_wn_simi_v12_|t_prob:0.35|top_k:10/i(46000)_epoch(7)_dev(0.6573657365736574)_loss(1.057304784975978)_seed(12)")
    # train_fever_v1_advsample_v2_shuffle_std_ema(resume_model=r_model)
    # train_fever_v1_advsample_v2_shuffle_std_ema_on_dev(resume_model=r_model)

    # train_fever_v1_advsample_v2_shuffle_bigger_train_on_dev()
    # train_fever_v1_advsample_shuffle_bigger()
    # train_fever_v1_advsample_shuffle_stad()
    # hidden_eval_fever_adv_v1()
    # hidden_eval_fever()
    # utest_data_loader()
    # spectrum_eval_manual_check()
    # train_fever_ema_v1()
    # train_fever_ema_v1_runtest()
    eval_m_on_sselection(
        "/home/easonnie/projects/FunEver/saved_models/11-23-23:45:19_bert_nsmn_ema_lr1|t_prob:0.02|top_k:8/i(104000)_epoch(15)_dev(0.6736173617361736)_lacc(0.7110711071107111)_seed(12)"
    )