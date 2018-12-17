import re
import math
import json
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': F.relu,
    'swish': swish,
    'gelu': gelu
}


def pad_1d(seq, pad_l):
    """
    The seq is a sequence having shape [T, ..]. Note: The seq contains only one instance. This is not batched.

    :param seq:  Input sequence with shape [T, ...]
    :param pad_l: The required pad_length.
    :return:  Output sequence will have shape [Pad_L, ...]
    """
    time = seq.size(0)
    if time >= pad_l:
        return seq[:pad_l, ]  # Truncate the length if the length is bigger than required padded_length.
    else:
        pad_seq = seq.new(pad_l - time, *seq.size()[1:]).zero_()  # Requires_grad is False
        return torch.cat([seq, pad_seq], dim=0)


def pack_sequence_for_linear(inputs, lengths):
    """
    :param inputs: [B, T, D]
    :param lengths:  [B]
    :return:    [sum(batch lengths), D]
    """
    batch_list = []
    for i, l in enumerate(lengths.tolist()):
        batch_list.append(inputs[i, :l])
    packed_sequence = torch.cat(batch_list, 0)

    return packed_sequence


def unpack_sequence_for_linear(inputs, lengths):
    batch_list = []
    max_l = max(lengths.tolist())

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    start = 0
    for l in lengths.tolist():
        end = start + l
        batch_list.append(pad_1d(inputs[start:end], max_l))
        start = end
    return torch.stack(batch_list, 0)


class SeqLayerNorm(nn.Module):
    "Sequential LayerNorm Layer."

    def __init__(self, n_state):
        super(SeqLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(n_state)

    def forward(self, seq, lengths):
        '''
        :param seq:     [B, T, D]
        :param lengths: [B]
        :return:        [B, T, D]
        '''
        packed_seq = pack_sequence_for_linear(seq, lengths)
        out_seq = unpack_sequence_for_linear(self.ln(packed_seq), lengths)
        return out_seq


# This is basically a affine layer
class Conv1D(nn.Module):
    def __init__(self, d_out, d_in):
        super(Conv1D, self).__init__()
        self.d_out = d_out

        w = torch.empty(d_in, d_out)
        b = torch.empty(d_out)
        # nn.init.normal_(w, std=0.02)
        nn.init.xavier_uniform_(w)
        nn.init.uniform_(b, -0.01, 0.01)
        self.w = Parameter(w)
        self.b = Parameter(b)

    def forward(self, x):
        '''
        :param x:   [B, T, D_in]
        :return:    [B, T, D_out]
        '''
        size_out = x.size()[:-1] + (self.d_out,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)  # (D_out) + (B * T, D_in) * (D_in, D_out)
        x = x.view(*size_out)  # (B, T, D_out)

        return x


class SeqMultiHeadAttention(nn.Module):
    def __init__(self, d_in, n_head, attn_drop_r, output_drop_r, scale=False):
        super(SeqMultiHeadAttention, self).__init__()
        # n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert d_in % n_head == 0
        # self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = d_in
        self.scale = scale
        self.input_to_q_k_v = Conv1D(d_in * 3, d_in)
        self.merge_head_proj = Conv1D(d_in, d_in)

        self.attn_dropout = nn.Dropout(attn_drop_r)
        self.output_dropout = nn.Dropout(output_drop_r)

    def _attn(self, q, k, v, q_lengths, k_lengths):
        '''
        :param q:   [B, d_head, T1, D_q]
        :param k:   [B, d_head D_k, T2] D_q = D_k
        :param v:   [B, d_head T2, D_v]
        :param lengths: [B]
        :return:    [B, d_head T1, D_v]
        '''
        w = torch.matmul(q, k)  # [B, d_head T1, T2]
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        # q_lengths, k_lengths -> attn_mask

        w_mask = w.new(*w.size()).fill_(0)  # [B, d_head, T1, T2]
        # Init similarity mask using lengths
        assert len(q_lengths) == len(k_lengths)
        assert int(w.size(0)) == len(k_lengths)
        for i, (l_1, l_2) in enumerate(zip(q_lengths, k_lengths)):
            w_mask[i][..., :l_1, :l_2] = 1
        attn_mask = w_mask
        w = w * attn_mask + -1e9 * (1 - attn_mask)  # TF implem method: mask_attn_weights
        # w = w + (- 1e9) * attn_mask  # TF implem method: mask_attn_weights
        # w.data.masked_fill_(attn_mask.byte(), -math.inf)

        w = nn.Softmax(dim=-1)(w)
        w = w * attn_mask
        w = self.attn_dropout(w)
        return torch.matmul(w, v)  # [B, d_head T1, T2] * [B, d_head T2, D_v] -> [B, d_head T1, D_v]

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, d_head T, D] -> [B, T, d_head, D]
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)  # [B, T, d_head * D]
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        # [B, T, d_head, D // d_head]
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # [B, d_head, D // d_head, T]
        else:
            return x.permute(0, 2, 1, 3)  # [B, d_head, T, D // d_head]

    def forward(self, x, lengths):
        x = self.input_to_q_k_v(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value, lengths, lengths)
        a = self.merge_heads(a)
        a = self.merge_head_proj(a)
        a = self.output_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, d_state, d_in, d_out, drop_r, activation_type):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        # Usually set d_state = 4 * d_in
        self.c_fc = Conv1D(d_state, d_in)
        self.c_proj = Conv1D(d_out, d_state)
        self.act = ACT_FNS[activation_type]
        self.dropout = nn.Dropout(drop_r)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, d_in, n_head, attn_drop_r, attn_output_drop_r,
                 block_mlp_drop_r, block_mlp_activation_type,
                 scale=False):
        super(Block, self).__init__()
        self.attn = SeqMultiHeadAttention(d_in, n_head, attn_drop_r, attn_output_drop_r, scale=scale)
        self.ln_1 = SeqLayerNorm(d_in)
        self.mlp = MLP(4 * d_in, d_in, d_in, block_mlp_drop_r, block_mlp_activation_type)
        self.ln_2 = SeqLayerNorm(d_in)

    def forward(self, x, lengths):
        a = self.attn(x, lengths)
        n = self.ln_1(x + a, lengths)
        m = self.mlp(n)
        h = self.ln_2(n + m, lengths)
        return h


class TransformerEncoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, n_block, d_in, n_head, attn_drop_r, attn_output_drop_r,
                 block_mlp_drop_r, block_mlp_activation_type,
                 scale=False):
        super(TransformerEncoder, self).__init__()
        block = Block(d_in, n_head, attn_drop_r, attn_output_drop_r,
                      block_mlp_drop_r, block_mlp_activation_type, scale)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(n_block)])

    def forward(self, seq, s_len):
        for block in self.h:
            seq = block(seq, s_len)
        return seq
