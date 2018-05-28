# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
from laed.utils import INT, FLOAT, LONG, cast_type
import numpy as np
import torch.nn.functional as F


def summary(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params
        # and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = summary(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ') Total Parameters={}'.format(total_params)
    return tmpstr


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for key, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def _gather_last_out(self, rnn_outs, lens):
        """
        :param rnn_outs: batch_size x T_len x dimension
        :param lens: [a list of lens]
        :return: batch_size x dimension
        """
        time_dimension = 1
        len_vars = self.np2var(np.array(lens), LONG)
        len_vars = len_vars.view(-1, 1).expand(len(lens), rnn_outs.size(2)).unsqueeze(1)
        slices = rnn_outs.gather(time_dimension, len_vars-1)
        return slices.squeeze(time_dimension)

    def _remove_padding(self, feats, words):
        """"
        :param feats: batch_size x num_words x feats
        :param words: batch_size x num_words
        :return: the same input without padding
        """
        if feats is None:
            return None, None

        batch_size = words.size(0)
        valid_mask = torch.sign(words).float()
        batch_lens = torch.sum(valid_mask, dim=1)
        max_word_num = torch.max(batch_lens)
        padded_lens = (max_word_num - batch_lens).cpu().data.numpy()
        valid_words = []
        valid_feats = []

        for b_id in range(batch_size):
            valid_idxs = valid_mask[b_id].nonzero().view(-1)
            valid_row_words = torch.index_select(words[b_id], 0, valid_idxs)
            valid_row_feat = torch.index_select(feats[b_id], 0, valid_idxs)

            padded_len = int(padded_lens[b_id])
            valid_row_words = F.pad(valid_row_words, (0, padded_len))
            valid_row_feat = F.pad(valid_row_feat, (0, 0, 0, padded_len))

            valid_words.append(valid_row_words.unsqueeze(0))
            valid_feats.append(valid_row_feat.unsqueeze(0))

        feats = torch.cat(valid_feats, dim=0)
        words = torch.cat(valid_words, dim=0)
        return feats, words

    def get_optimizer(self, config):
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr)
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
