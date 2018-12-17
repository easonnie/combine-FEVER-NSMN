import torch
from torch import nn
import torch.nn.functional as F
import math


class biDafAttn(nn.Module):
    def __init__(self, channel_size):
        super(biDafAttn, self).__init__()
        """
        This method do biDaf from s2 to s1:
            The return value will have the same size as s1.
        :param channel_size: Hidden size of the input
        """

    def similarity(self, s1, l1, s2, l2):
        """
        :param s1: [B, t1, D]
        :param l1: [B]
        :param s2: [B, t2, D]
        :param l2: [B]
        :return:
        """
        batch_size = s1.size(0)
        t1 = s1.size(1)
        t2 = s2.size(1)
        S = torch.bmm(s1, s2.transpose(1, 2))
        # [B, t1, D] * [B, D, t2] -> [B, t1, t2] S is the similarity matrix from biDAF paper. [B, T1, T2]

        s_mask = S.data.new(*S.size()).fill_(1).byte()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(l1, l2)):
            s_mask[i][:l_1, :l_2] = 0

        S.data.masked_fill_(s_mask.byte(), -math.inf)
        return S

    def get_U_tile(self, S, s2):
        a_weight = F.softmax(S, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        U_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]
        return U_tile

    def get_both_tile(self, S, s1, s2):
        a_weight = F.softmax(S, dim=2)  # [B, t1, t2]
        a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)  # remove nan from softmax on -inf
        U_tile = torch.bmm(a_weight, s2)  # [B, t1, t2] * [B, t2, D] -> [B, t1, D]

        a1_weight = F.softmax(S, dim=1)  # [B, t1, t2]
        a1_weight.data.masked_fill_(a1_weight.data != a1_weight.data, 0)  # remove nan from softmax on -inf
        U1_tile = torch.bmm(a1_weight.transpose(1, 2), s1)  # [B, t2, t1] * [B, t1, D] -> [B, t2, D]
        return U_tile, U1_tile

    def forward(self, s1, l1, s2, l2):
        S = self.similarity(s1, l1, s2, l2)
        U_tile = self.get_U_tile(S, s2)
        return U_tile