# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import print_function
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
import torch
from laed.utils import INT, FLOAT, LONG, cast_type
import logging


class L2Loss(_Loss):

    logger = logging.getLogger()
    def forward(self, state_a, state_b):
        if type(state_a) is tuple:
            losses = 0.0
            for s_a, s_b in zip(state_a, state_b):
                losses += torch.pow(s_a-s_b, 2)
        else:
            losses = torch.pow(state_a-state_b, 2)
        return torch.mean(losses)

class NLLEntropy(_Loss):

    logger = logging.getLogger()
    def __init__(self, padding_idx, config, rev_vocab=None, key_vocab=None):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = config.avg_type

        if rev_vocab is None or key_vocab is None:
            self.weight = None
        else:
            self.logger.info("Use extra cost for key words")
            weight = np.ones(len(rev_vocab))
            for key in key_vocab:
                weight[rev_vocab[key]] = 10.0
            self.weight = cast_type(torch.from_numpy(weight), FLOAT,
                                    config.use_gpu)

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        if self.avg_type is None:
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss/word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        else:
            raise ValueError("Unknown avg type")

        return loss


class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
        loss -= torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * torch.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * torch.sum(loss, dim=1)
        avg_kl_loss = torch.mean(kl_loss)
        return avg_kl_loss


class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()

        qy = torch.exp(log_qy)
        y_kl = torch.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return torch.mean(y_kl)
        else:
            return torch.sum(y_kl)/batch_size


class CrossEntropyoss(_Loss):
    def __init__(self):
        super(CrossEntropyoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        -qy log(qy) + qy * log(q(y)/p(y))
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        kl_qp = torch.sum(qy * (log_qy - log_py), dim=1)
        cross_ent = h_q + kl_qp
        if unit_average:
            return torch.mean(cross_ent)
        else:
            return torch.sum(cross_ent)/batch_size


class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = torch.exp(log_qy)
        h_q = torch.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return torch.mean(h_q)
        else:
            return torch.sum(h_q) / batch_size




