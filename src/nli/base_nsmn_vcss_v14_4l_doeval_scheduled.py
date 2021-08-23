# This method is created on 25 Nov 2018 11:16, aiming for NAACL.
# This code is used to train a single model for both verification and selection.
# The code is modified from nsmn mesim_wn_simi_v1_2 and nsmn_sent_wise_v1_1

# The code is modified from base_nsmn_vcss_v11 to 4 logits output that do one pass only on support and refute examples.
import datetime
import random
import math
from collections import defaultdict

import torch
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.modules import Embedding, Elmo

from data_util.data_readers.fever_reader_joint_ss_vc import VCSS_Reader
from data_util.data_readers.fever_reader_with_wn_simi import WNSIMIReader
from neural_modules.ema import EMA
from sample_for_nli.adv_sampler_v01 import get_adv_sampled_data
from sentence_retrieval.nn_postprocess_ablation import score_converter_scaled
from torch import nn
import copy
from neural_modules.ema import load_ema_to_model, save_ema_to_file
from neural_modules.loss_tool import log_value_recover
import os

import config

from data_util.exvocab import load_vocab_embeddings
from sentence_retrieval.sampler_for_nmodel import get_full_list
from simi_sampler_nli_v0.simi_sampler import paired_selection_score_dict, threshold_sampler, \
    select_sent_with_prob_for_eval, adv_simi_sample_with_prob_v1_0, adv_simi_sample_with_prob_v1_1, \
    select_sent_with_prob_for_eval_list, threshold_sampler_insure_unique
from utils import common

from log_util import save_tool

from flint import torch_util
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from neural_modules import biDafAttn
from utils import c_scorer
from vc_ss_naacl.data_wrangler import VCSSTrainingSampler
from wn_featurizer import wn_persistent_api
import data_util.paragraph_span as span_tool
import vc_ss_naacl as vc_ss

from pathlib import Path

support_index = 0
refute_index = 1
nei_index = 2
selection_index = 3
non_selection_index = 4


class ESIM(nn.Module):
    # This is ESIM sequence matching model
    # lstm
    def __init__(self, rnn_size_in=(1024 + 300, 1024 + 300), rnn_size_out=(300, 300), max_l=100,
                 mlp_d=300, num_of_class=4, drop_r=0.5, activation_type='relu'):
        # change number of class to 4 so that we have 4 logits.

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


def compute_mixing_loss(model, out, batch, criterion, data_sampler: vc_ss.data_wrangler.VCSSTrainingSampler=None):
    batch_size = out.size(0)

    vc_batch_list = []
    vc_y_list = []

    ss_batch_list = []  # SS batch can be just concatenated bc it need only one batch dimension
    ss_y_list = []

    ir_index = 3    # irrelevant index is three. 0,1,2,3
    ir_prob_out = torch.sigmoid(out[:, ir_index])

    # The four value are used in this way:
    # The first three label are used for Supports, Refutes, or Not Enough Info.
    # The last value is used for evidence retrieval.
    # When we train the evidence retrieval,
    # we use the negative example randomly sampled from upstream document.
    # we use the positive example which is the ground truth evidence by the annotator.

    # [0, 1, 2]
    # torch.sigmoid on out[i][3] indicate the prob of irrelevant.

    # f_out = F.softmax(out, dim=1)

    for i in range(batch_size):
        pid = batch['pid'][i]

        if c_scorer.SENT_LINE in pid: # SS
            # prob_values = torch.stack([1 - f_out[i][2], f_out[i][2]])
            # zero_count = int(torch.sum(prob_values == 0))
            # if zero_count >= 1:
            #     prob_values = prob_values + 1e-28   # add small value to prevent gradient problem.

            ss_batch_list.append(out[i][ir_index])
            cur_y = (batch['label'][i] - 3).float().to(next(model.parameters()).device)
            ss_y_list.append(cur_y)
            # 0 indicates selection
            # 1 indicates non-selection
            # Remember

            # Compute the score on the fly
            if data_sampler is not None:
                score_s = float(out[i][0])
                score_r = float(out[i][1])
                score_nei = float(out[i][2])

                ir_score = float(out[i][ir_index])
                ir_prob = float(ir_prob_out[i])

                total = math.exp(score_s) + math.exp(score_r) + math.exp(score_nei)
                prob_s = math.exp(score_s) / total
                prob_r = math.exp(score_r) / total
                prob_nei = math.exp(score_nei) / total

                item = {
                    'score_s': score_s,
                    'score_r': score_r,
                    'score_nei': score_nei,
                    'prob_s': prob_s,
                    'prob_r': prob_r,
                    'prob_nei': prob_nei,

                    'score': - ir_score,  # important value, remember to set this to negative
                    'prob': 1 - ir_prob,    # important value
                }

                data_sampler.assign_score_direct(pid, item)

        else:                           # VC
            vc_batch_list.append(out[i][:3])  # We only need the first three
            vc_y_list.append(batch['label'][i])

            if batch['label'][i] == support_index or batch['label'][i] == refute_index:
                # if the label is support or refute,
                # the current example should be treated as selection and use to train IR
                ss_batch_list.append(out[i][ir_index])  # this is zero dimension
                ss_y_list.append(out[i][ir_index].new_full(out[i][ir_index].size(), 0))
                # 0 indicate selection and 1 indicate non-selection, remember!

    if len(vc_batch_list) == 0: # all ss
        ss_out = torch.stack(ss_batch_list, dim=0)
        ss_y = torch.stack(ss_y_list, dim=0)
        ss_y = ss_y.to(next(model.parameters()).device)

        # return F.nll_loss(torch.log(ss_out), ss_y)
        return F.binary_cross_entropy_with_logits(ss_out, ss_y)
        # return criterion(ss_out, ss_y)

    elif len(ss_batch_list) == 0:   # all vc
        vc_out = torch.stack(vc_batch_list, dim=0)
        vc_y = torch.stack(vc_y_list, dim=0)
        assert torch.equal(vc_y, batch['label'])
        vc_y = vc_y.to(next(model.parameters()).device)

        return criterion(vc_out, vc_y)

    else:   # ss + vc
        ss_out = torch.stack(ss_batch_list, dim=0)
        ss_y = torch.stack(ss_y_list, dim=0)
        ss_y = ss_y.to(next(model.parameters()).device)

        vc_out = torch.stack(vc_batch_list, dim=0)
        vc_y = torch.stack(vc_y_list, dim=0)
        vc_y = vc_y.to(next(model.parameters()).device)

        return criterion(vc_out, vc_y) + F.binary_cross_entropy_with_logits(ss_out, ss_y)
        # return criterion(vc_out, vc_y) + criterion(ss_out, ss_y)


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


def hidden_eval_vc(model, data_iter, dev_data_list, with_logits=False, with_probs=False):
    # SUPPORTS < (-.-) > 0
    # REFUTES < (-.-) > 1
    # NOT ENOUGH INFO < (-.-) > 2

    print("Eval VC", end=' ')
    print(datetime.datetime.now())

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

        # for batch_idx, batch in enumerate(tqdm(data_iter)):
        for batch_idx, batch in enumerate(data_iter):
            full_out = model(batch)

            out = full_out[:, :3]
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


def hidden_eval_ss(model, data_iter, dev_data_list):
    # select < (-.-) > 0
    # non-select < (-.-) > 1
    # hidden < (-.-) > -2
    print("Eval SS", end=' ')
    print(datetime.datetime.now())

    with torch.no_grad():
        id2label = {
            0: "true",
            1: "false",
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

        y_ir_logits_list = []
        y_ir_prob_list = []

        y_id_list = []

        # for batch_idx, batch in enumerate(tqdm(data_iter)):
        for batch_idx, batch in enumerate(data_iter):
            full_out = model(batch)
            out = full_out[:, :3]
            prob = F.softmax(out, dim=1)

            ir_out = full_out[:, 3]
            ir_prob = torch.sigmoid(ir_out)

            y = batch['label']
            y_id_list.extend(list(batch['pid']))

            y_s_logits_list.extend(out[:, 0].tolist())
            y_s_prob_list.extend(prob[:, 0].tolist())

            y_r_logits_list.extend(out[:, 1].tolist())
            y_r_prob_list.extend(prob[:, 1].tolist())

            y_nei_logits_list.extend(out[:, 2].tolist())
            y_nei_prob_list.extend(prob[:, 2].tolist())

            y_ir_logits_list.extend(ir_out[:].tolist())
            y_ir_prob_list.extend(ir_prob[:].tolist())

            totoal_size += y.size(0)

        assert len(y_id_list) == len(dev_data_list)
        assert len(y_s_logits_list) == len(dev_data_list)
        assert len(y_r_logits_list) == len(dev_data_list)
        assert len(y_nei_logits_list) == len(dev_data_list)
        assert len(y_ir_logits_list) == len(dev_data_list)
        assert len(y_ir_prob_list) == len(dev_data_list)

        for i in range(len(dev_data_list)):
            assert str(y_id_list[i]) == str(dev_data_list[i]['selection_id'])
            # if math.fabs(1 - (y_s_prob_list[i] + y_r_prob_list[i] + y_nei_prob_list[i])) >= 1e-3:

            # print(y_s_prob_list[i], y_r_prob_list[i], y_nei_prob_list[i])
            # print(y_s_logits_list[i], y_r_logits_list[i], y_nei_logits_list[i])
            # print(out[i])
            # print(prob[i])
            # print(math.fabs(1 - (y_s_prob_list[i] + y_r_prob_list[i] + y_nei_prob_list[i])))

            assert math.fabs(1 - (y_s_prob_list[i] + y_r_prob_list[i] + y_nei_prob_list[i])) <= 1e-3
            # Matching id

            dev_data_list[i]['score_s'] = y_s_logits_list[i]
            dev_data_list[i]['prob_s'] = y_s_prob_list[i]

            dev_data_list[i]['score_r'] = y_r_logits_list[i]
            dev_data_list[i]['prob_r'] = y_r_prob_list[i]

            dev_data_list[i]['score_nei'] = y_nei_logits_list[i]
            dev_data_list[i]['prob_nei'] = y_nei_prob_list[i]

            item = dev_data_list[i]
            s_r = math.exp(item['score_s']) + math.exp(item['score_r'])
            nei = math.exp(item['score_nei'])
            total = s_r + nei
            assert (nei / total) - (item['prob_nei']) < 1e-3

            # We might need to pos modify this scoring function for better performance.
            dev_data_list[i]['score'] = - y_ir_logits_list[i]
            dev_data_list[i]['prob'] = 1 - y_ir_prob_list[i]
            # Reset neural set

        print('total_size:', totoal_size)

        return dev_data_list


def eval_sent_for_sampler(model, token_indexers, vocab, vc_ss_training_sampler):
    max_len = 80
    batch_size = 64
    data_reader = VCSS_Reader(token_indexers=token_indexers, lazy=True, max_l=max_len)

    data_list = vc_ss_training_sampler.sent_list
    vc_ss.data_wrangler.assign_task_label(data_list, 'ss')
    print("Whole training size:", len(data_list))

    data_instance = data_reader.read(data_list)

    biterator = BasicIterator(batch_size=batch_size)
    biterator.index_with(vocab)

    data_iter = biterator(data_instance, shuffle=False, num_epochs=1)

    print("Eval for whole training set")
    print(datetime.datetime.now())

    with torch.no_grad():
        # id2label = {
        #     0: "true",
        #     1: "false",
        #     -2: "hidden"
        # }

        # print("Evaluating ...")
        model.eval()

        for batch_idx, batch in enumerate(data_iter):
            full_out = model(batch)
            # out = full_out[:, :3]
            # prob = F.softmax(out, dim=1)

            the_batch_size = full_out.size(0)

            ir_index = 3  # irrelevant index is three. 0,1,2,3
            ir_prob_out = torch.sigmoid(full_out[:, ir_index])

            y = batch['label']

            for i in range(the_batch_size):
                pid = batch['pid'][i]

                score_s = float(full_out[i][0])
                score_r = float(full_out[i][1])
                score_nei = float(full_out[i][2])

                ir_score = float(full_out[i][ir_index])
                ir_prob = float(ir_prob_out[i])

                total = math.exp(score_s) + math.exp(score_r) + math.exp(score_nei)
                prob_s = math.exp(score_s) / total
                prob_r = math.exp(score_r) / total
                prob_nei = math.exp(score_nei) / total

                item = {
                    'score_s': score_s,
                    'score_r': score_r,
                    'score_nei': score_nei,
                    'prob_s': prob_s,
                    'prob_r': prob_r,
                    'prob_nei': prob_nei,

                    'score': - ir_score,  # important value, remember to set this to negative
                    'prob': 1 - ir_prob,  # important value
                }

                vc_ss_training_sampler.assign_score_direct(pid, item)

            if batch_idx % 10000 == 0:
                print(batch_idx, end=' ')
                print(datetime.datetime.now())

    print()


def train_fever_std_ema_v1(resume_model=None, do_analysis=False):
    """
    This method is created on 26 Nov 2018 08:50 with the purpose of training vc and ss all together.
    :param resume_model:
    :param wn_feature:
    :return:
    """

    num_epoch = 200
    seed = 12
    batch_size = 32
    lazy = True
    train_prob_threshold = 0.02
    train_sample_top_k = 8
    dev_prob_threshold = 0.1
    dev_sample_top_k = 5

   #  neg_sample_upper_prob = 0.1
    # decay_r = 0.01
    schedule_sample_dict = defaultdict(lambda: 0.0025)

    schedule_sample_dict.update({
        0: 0.1, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.0025
    })

    neg_only = False

    top_k_doc = 5

    debug = False

    eval_epoch_num = [2, 4]

    experiment_name = f"nsmn_vc_ss_std_ema_lr1_v12_negpos_doeval:{neg_only}|t_prob:{train_prob_threshold}|top_k:{train_sample_top_k}_scheduled_neg_sampler"
    # resume_model = None

    print("Do EMA:")

    print("Dev prob threshold:", dev_prob_threshold)
    print("Train prob threshold:", train_prob_threshold)
    print("Train sample top k:", train_sample_top_k)

    # Get upstream sentence document retrieval data
    dev_doc_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl"
    train_doc_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/train_doc.jsonl"

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_doc_upstream_file, pred=True,
                                               top_k=top_k_doc)

    complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_doc_upstream_file, pred=False,
                                                 top_k=top_k_doc)
    if debug:
        complete_upstream_dev_data = complete_upstream_dev_data[:1000]
        complete_upstream_train_data = complete_upstream_train_data[:1000]

    print("Dev size:", len(complete_upstream_dev_data))
    print("Train size:", len(complete_upstream_train_data))

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    # Data Reader
    dev_fever_data_reader = VCSS_Reader(token_indexers=token_indexers, lazy=lazy, max_l=260)
    train_fever_data_reader = VCSS_Reader(token_indexers=token_indexers, lazy=lazy, max_l=260)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")

    vocab.add_token_to_namespace('true', namespace='labels')
    vocab.add_token_to_namespace('false', namespace='labels')
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    # Reader and prepare end

    vc_ss_training_sampler = VCSSTrainingSampler(complete_upstream_train_data)
    vc_ss_training_sampler.show_info()

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + 1,
                               1024 + 450 + 1),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300, num_of_class=4)

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

    analysis_dir = None
    if do_analysis:
        analysis_dir = Path(file_path_prefix) / "analysis_aux"
        analysis_dir.mkdir()
    # Save source code end.

    # Staring parameter setup
    best_dev = -1
    iteration = 0

    start_lr = 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()
    # parameter setup end

    for i_epoch in range(num_epoch):
        print("Resampling...")
        # This is for train
        # This is for sample candidate data for from result of ss for vc.
        # This we will need to do after each epoch.
        if i_epoch in eval_epoch_num:
            print("We now need to eval the whole training set.")
            print("Be patient and hope good luck!")
            load_ema_to_model(cloned_empty_model, ema)
            eval_sent_for_sampler(cloned_empty_model, token_indexers, vocab, vc_ss_training_sampler)

        train_data_with_candidate_sample_list = vc_ss.data_wrangler.sample_sentences_for_vc_with_nei(
            config.T_FEVER_TRAIN_JSONL, vc_ss_training_sampler.sent_list,
            train_prob_threshold, train_sample_top_k)   # We initialize the prob for each sentence so the sampler can work, but we will need to run the model for dev data to work.

        train_selection_dict = paired_selection_score_dict(vc_ss_training_sampler.sent_list)

        cur_train_vc_data = adv_simi_sample_with_prob_v1_1(config.T_FEVER_TRAIN_JSONL,
                                                           train_data_with_candidate_sample_list,
                                                           train_selection_dict,
                                                           tokenized=True)

        if do_analysis:
            # Customized analysis output
            common.save_jsonl(
                vc_ss_training_sampler.sent_list,
                analysis_dir / f"E_{i_epoch}_whole_train_sent_{save_tool.get_cur_time_str()}.jsonl")
            common.save_jsonl(
                train_data_with_candidate_sample_list,
                analysis_dir / f"E_{i_epoch}_sampled_train_sent_{save_tool.get_cur_time_str()}.jsonl")
            common.save_jsonl(
                cur_train_vc_data,
                analysis_dir / f"E_{i_epoch}_train_vc_data_{save_tool.get_cur_time_str()}.jsonl")

        print(f"E{i_epoch} VC_data:", len(cur_train_vc_data))

        # This is for sample negative candidate data for ss
        # After sampling, we decrease the ratio.
        neg_sample_upper_prob = schedule_sample_dict[i_epoch]
        print("Neg Sampler upper rate:", neg_sample_upper_prob)
        # print("Rate decreasing")
        # neg_sample_upper_prob -= decay_r
        neg_sample_upper_prob = max(0.000, neg_sample_upper_prob)

        cur_train_ss_data = vc_ss_training_sampler.sample_for_ss(neg_only=neg_only, upper_prob=neg_sample_upper_prob)
        vc_ss_training_sampler.show_info(cur_train_ss_data)
        print(f"E{i_epoch} SS_data:", len(cur_train_ss_data))


        vc_ss.data_wrangler.assign_task_label(cur_train_ss_data, 'ss')
        vc_ss.data_wrangler.assign_task_label(cur_train_vc_data, 'vc')

        vs_ss_train_list = cur_train_ss_data + cur_train_vc_data
        random.shuffle(vs_ss_train_list)
        print(f"E{i_epoch} Total ss+vc:", len(vs_ss_train_list))
        vc_ss_instance = train_fever_data_reader.read(vs_ss_train_list)

        train_iter = biterator(vc_ss_instance, shuffle=True, num_epochs=1)

        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            loss = compute_mixing_loss(model, out, batch, criterion, vc_ss_training_sampler)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            # EMA update
            ema(model.named_parameters())

            if i_epoch < 10:
                mod = 20000
                # mod = 100
            else:
                mod = 2000

            if iteration % mod == 0:

                # This is the code for eval:
                load_ema_to_model(cloned_empty_model, ema)

                vc_ss.data_wrangler.assign_task_label(complete_upstream_dev_data, 'ss')
                dev_ss_instance = dev_fever_data_reader.read(complete_upstream_dev_data)
                eval_ss_iter = biterator(dev_ss_instance, num_epochs=1, shuffle=False)
                scored_dev_sent_data = hidden_eval_ss(cloned_empty_model, eval_ss_iter, complete_upstream_dev_data)

                # for vc
                filtered_dev_list = vc_ss.data_wrangler.sample_sentences_for_vc_with_nei(config.T_FEVER_DEV_JSONL,
                                                                                         scored_dev_sent_data,
                                                                                         dev_prob_threshold,
                                                                                         dev_sample_top_k)

                dev_selection_dict = paired_selection_score_dict(scored_dev_sent_data)
                ready_dev_list = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, filtered_dev_list,
                                                                dev_selection_dict, tokenized=True)

                vc_ss.data_wrangler.assign_task_label(ready_dev_list, 'vc')
                dev_vc_instance = dev_fever_data_reader.read(ready_dev_list)
                eval_vc_iter = biterator(dev_vc_instance, num_epochs=1, shuffle=False)
                eval_dev_result_list = hidden_eval_vc(cloned_empty_model, eval_vc_iter, ready_dev_list)

                # Scoring
                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(eval_dev_result_list,
                                                                            common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                            mode=eval_mode,
                                                                            verbose=False)
                print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

                print(f"Dev:{strict_score}/{acc_score}")

                if do_analysis:
                    # Customized analysis output
                    common.save_jsonl(
                        scored_dev_sent_data,
                        analysis_dir / f"E_{i_epoch}_scored_dev_sent_{save_tool.get_cur_time_str()}.jsonl")
                    common.save_jsonl(
                        eval_dev_result_list,
                        analysis_dir / f"E_{i_epoch}_eval_vc_output_data_{save_tool.get_cur_time_str()}.jsonl")

                need_save = False
                if strict_score > best_dev:
                    best_dev = strict_score
                    need_save = True

                if need_save:
                    # save_path = os.path.join(
                    #     file_path_prefix,
                    #     f'i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_seed({seed})'
                    # )

                    # torch.save(model.state_dict(), save_path)

                    ema_save_path = os.path.join(
                        file_path_prefix,
                        f'ema_i({iteration})_epoch({i_epoch})_dev({strict_score})_lacc({acc_score})_p({pr})_r({rec})_f1({f1})_seed({seed})')

                    save_ema_to_file(ema, ema_save_path)


def analysis_model(model_path):
    batch_size = 32
    lazy = True
    train_prob_threshold = 0.02
    train_sample_top_k = 8
    dev_prob_threshold = 0.1
    dev_sample_top_k = 5

    neg_sample_upper_prob = 0.006
    decay_r = 0.002

    top_k_doc = 5
    dev_doc_upstream_file = config.RESULT_PATH / "doc_retri/std_upstream_data_using_pageview/dev_doc.jsonl"

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_doc_upstream_file, pred=True,
                                               top_k=top_k_doc)

    print("Dev size:", len(complete_upstream_dev_data))

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    # Data Reader
    dev_fever_data_reader = VCSS_Reader(token_indexers=token_indexers, lazy=lazy, max_l=260)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")

    vocab.add_token_to_namespace('true', namespace='labels')
    vocab.add_token_to_namespace('false', namespace='labels')
    vocab.add_token_to_namespace("hidden", namespace="labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='labels')

    print(vocab.get_token_to_index_vocabulary('labels'))
    print(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    # Reader and prepare end

    # vc_ss_training_sampler = VCSSTrainingSampler(complete_upstream_train_data)
    # vc_ss_training_sampler.show_info()

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + 1,
                               1024 + 450 + 1),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    print("Model Max length:", model.max_l)

    model.display()
    model.to(device)

    cloned_empty_model = copy.deepcopy(model)

    load_ema_to_model(cloned_empty_model, model_path)

    vc_ss.data_wrangler.assign_task_label(complete_upstream_dev_data, 'ss')
    dev_ss_instance = dev_fever_data_reader.read(complete_upstream_dev_data)
    eval_ss_iter = biterator(dev_ss_instance, num_epochs=1, shuffle=False)
    scored_dev_sent_data = hidden_eval_ss(cloned_empty_model, eval_ss_iter, complete_upstream_dev_data)

    common.save_jsonl(scored_dev_sent_data, "dev_scored_sent_data.jsonl")
    # for vc
    filtered_dev_list = vc_ss.data_wrangler.sample_sentences_for_vc_with_nei(config.T_FEVER_DEV_JSONL,
                                                                             scored_dev_sent_data,
                                                                             dev_prob_threshold,
                                                                             dev_sample_top_k)
    common.save_jsonl(filtered_dev_list, "dev_scored_sent_data_after_sample.jsonl")

    dev_selection_dict = paired_selection_score_dict(scored_dev_sent_data)
    ready_dev_list = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, filtered_dev_list,
                                                    dev_selection_dict, tokenized=True)

    vc_ss.data_wrangler.assign_task_label(ready_dev_list, 'vc')
    dev_vc_instance = dev_fever_data_reader.read(ready_dev_list)
    eval_vc_iter = biterator(dev_vc_instance, num_epochs=1, shuffle=False)
    eval_dev_result_list = hidden_eval_vc(cloned_empty_model, eval_vc_iter, ready_dev_list)

    common.save_jsonl(eval_dev_result_list, "dev_nli_results.jsonl")

    # Scoring
    eval_mode = {'check_sent_id_correct': True, 'standard': True}
    strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(eval_dev_result_list,
                                                                common.load_jsonl(config.T_FEVER_DEV_JSONL),
                                                                mode=eval_mode,
                                                                verbose=False)
    print("Fever Score(Strict/Acc./Precision/Recall/F1):", strict_score, acc_score, pr, rec, f1)

    print(f"Dev:{strict_score}/{acc_score}")


def append_hidden_label(d_list):
    for item in d_list:
        item['label'] = 'hidden'
    return d_list



if __name__ == "__main__":
    # train_fever()
    # train_fever_v1_advsample()
    train_fever_std_ema_v1(do_analysis=True)
    # analysis_model("/home/easonnie/projects/FunEver/saved_models/11-28-11:27:44_nsmn_vc_ss_std_ema_lr1|t_prob:0.02|top_k:8_neg_sampler_r:0.015_dr:0.0005/ema_i(10000)_epoch(1)_dev(0.34493449344934496)_lacc(0.3511851185118512)_p(0.9862223722372238)_r(0.05378037803780378)_f1(0.10199859951184036)_seed(12)")
    # "/home/easonnie/projects/FunEver/saved_models/11-28-11:27:44_nsmn_vc_ss_std_ema_lr1|t_prob:0.02|top_k:8_neg_sampler_r:0.015_dr:0.0005/ema_i(10000)_epoch(1)_dev(0.34493449344934496)_lacc(0.3511851185118512)_p(0.9862223722372238)_r(0.05378037803780378)_f1(0.10199859951184036)_seed(12)"
    # pass
