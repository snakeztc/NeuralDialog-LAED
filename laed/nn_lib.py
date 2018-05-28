# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
import torch
import torch.nn as nn
import torch.nn.functional as F
from laed.utils import FLOAT, LONG, cast_type
from torch.autograd import Variable, Function


class CatConnector(nn.Module):
    def __init__(self, rnn_cell, hidden_size, input_size,
                 output_size, activation_fn=None):

        super(CatConnector, self).__init__()

        if rnn_cell == 'lstm':
            self.fch = nn.Linear(hidden_size+input_size, output_size)
            self.fcc = nn.Linear(hidden_size+input_size, output_size)
        else:
            self.fc = nn.Linear(hidden_size+input_size, output_size)

        self.activation_fn = activation_fn
        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, hidden_state, inputs):
        """
        :param hidden_state: [num_layer, batch_size, feat_size]
        :param inputs: [batch_size, feat_size]
        :return: 
        """
        if self.rnn_cell == 'lstm':
            h, c = hidden_state
            num_layer = h.size()[0]
            new_h = self.fch(torch.cat([h.view(-1, self.hidden_size), inputs], 1))
            new_c = self.fch(torch.cat([c.view(-1, self.hidden_size), inputs], 1))
            return (new_h.view(num_layer, -1, self.output_size),
                    new_c.view(num_layer, -1, self.output_size))
        else:
            num_layer = hidden_state.size()[0]
            new_s = self.fc(torch.cat([hidden_state.view(-1, self.hidden_size),
                                       inputs]))
            new_s = new_s.view(num_layer, -1, self.output_size)
            return new_s


class Bi2UniConnector(nn.Module):
    def __init__(self, rnn_cell, num_layer, hidden_size, output_size):

        super(Bi2UniConnector, self).__init__()
        if rnn_cell == 'lstm':
            self.fch = nn.Linear(hidden_size*2*num_layer, output_size)
            self.fcc = nn.Linear(hidden_size*2*num_layer, output_size)
        else:
            self.fc = nn.Linear(hidden_size*2*num_layer, output_size)

        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, hidden_state):
        """
        :param hidden_state: [num_layer, batch_size, feat_size]
        :param inputs: [batch_size, feat_size]
        :return: 
        """
        if self.rnn_cell == 'lstm':
            h, c = hidden_state
            num_layer = h.size()[0]
            flat_h = h.transpose(0, 1).contiguous()
            flat_c = c.transpose(0, 1).contiguous()
            new_h = self.fch(flat_h.view(-1, self.hidden_size*num_layer))
            new_c = self.fch(flat_c.view(-1, self.hidden_size*num_layer))
            return (new_h.view(1, -1, self.output_size),
                    new_c.view(1, -1, self.output_size))
        else:
            num_layer = hidden_state.size()[0]
            new_s = self.fc(hidden_state.view(-1, self.hidden_size*num_layer))
            new_s = new_s.view(1, -1, self.output_size)
            return new_s


class IdentityConnector(nn.Module):
    def __init__(self):
        super(IdentityConnector, self).__init__()

    def forward(self, hidden_state):
        return hidden_state


class AttnConnector(nn.Module):
    def __init__(self, rnn_cell, query_size, key_size, content_size,
                 output_size, attn_size):

        super(AttnConnector, self).__init__()

        self.query_embed = nn.Linear(query_size, attn_size)
        self.key_embed = nn.Linear(key_size, attn_size)
        self.attn_w = nn.Linear(attn_size, 1)

        if rnn_cell == 'lstm':
            self.project_h = nn.Linear(content_size+query_size, output_size)
            self.project_c = nn.Linear(content_size+query_size, output_size)
        else:
            self.project = nn.Linear(content_size+query_size, output_size)

        self.rnn_cell = rnn_cell
        self.query_size = query_size
        self.key_size = key_size
        self.content_size = content_size
        self.output_size = output_size

    def forward(self, queries, keys, contents):
        batch_size = keys.size(0)
        num_key = keys.size(1)

        query_embeded = self.query_embed(queries)
        key_embeded = self.key_embed(keys)

        tiled_query = query_embeded.unsqueeze(1).repeat(1, num_key, 1)
        fc1 = F.tanh(tiled_query + key_embeded)
        attn = self.attn_w(fc1).squeeze(-1)
        attn = F.sigmoid(attn.view(-1, num_key)).view(batch_size, -1, num_key)
        mix = torch.bmm(attn, contents).squeeze(1)
        out = torch.cat([mix, queries], dim=1)

        if self.rnn_cell == 'lstm':
            h = self.project_h(out).unsqueeze(0)
            c = self.project_c(out).unsqueeze(0)
            new_s = (h, c)
        else:
            new_s = self.project(out).unsqueeze(0)

        return new_s


class LinearConnector(nn.Module):
    def __init__(self, input_size, output_size, is_lstm, has_bias=True):
        super(LinearConnector, self).__init__()
        if is_lstm:
            self.linear_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.linear_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size 
        :return: 
        """
        if self.is_lstm:
            h = self.linear_h(inputs).unsqueeze(0)
            c = self.linear_c(inputs).unsqueeze(0)
            return (h, c)
        else:
            return self.linear(inputs).unsqueeze(0)

    def get_w(self):
        if self.is_lstm:
            return self.linear_h.weight
        else:
            return self.linear.weight


class Hidden2Feat(nn.Module):
    def __init__(self, input_size, output_size, is_lstm, has_bias=True):
        super(Hidden2Feat, self).__init__()
        if is_lstm:
            self.linear_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.linear_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.linear = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h = self.linear_h(inputs[0].squeeze(0))
            c = self.linear_c(inputs[1].squeeze(0))
            return h+c
        else:
            return self.linear(inputs.squeeze(0))


class Hidden2Gaussian(nn.Module):
    def __init__(self, input_size, output_size, is_lstm=False, has_bias=True):
        super(Hidden2Gaussian, self).__init__()
        if is_lstm:
            self.mu_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_h = nn.Linear(input_size, output_size, bias=has_bias)

            self.mu_c = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.mu = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar = nn.Linear(input_size, output_size, bias=has_bias)

        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c= inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)

            mu_h, mu_c = self.mu_h(h), self.mu_c(c)
            logvar_h, logvar_c = self.logvar_h(h), self.logvar_c(c)
            return mu_h+mu_c, logvar_h+logvar_c
        else:
            if inputs.dim() == 3:
                inputs = inputs.squeeze(0)

            mu = self.mu(inputs.squeeze(0))
            logvar = self.logvar(inputs.squeeze(0))
            return mu, logvar


class GumbelConnector(nn.Module):
    def __init__(self):
        super(GumbelConnector, self).__init__()

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature, use_gpu, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, use_gpu)
        _, y_hard = torch.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y


class GreedyConnector(nn.Module):
    def __init__(self):
        super(GreedyConnector, self).__init__()

    def forward(self, logits, use_gpu, return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        _, y_hard = torch.max(logits, dim=1, keepdim=True)
        y_onehot = cast_type(Variable(torch.zeros(logits.size())), FLOAT, use_gpu)
        y_onehot.scatter_(1, y_hard, 1.0)
        y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y


class GaussianConnector(nn.Module):
    def __init__(self):
        super(GaussianConnector, self).__init__()

    def forward(self, mu, logvar, use_gpu):
        """
        Sample a sample from a multivariate Gaussian distribution with a diagonal covariance matrix using the
        reparametrization trick.

        TODO: this should be better be a instance method in a Gaussian class.

        :param mu: a tensor of size [batch_size, variable_dim]. Batch_size can be None to support dynamic batching
        :param logvar: a tensor of size [batch_size, variable_dim]. Batch_size can be None.
        :return:
        """
        epsilon = torch.randn(logvar.size())
        epsilon = cast_type(Variable(epsilon), FLOAT, use_gpu)
        std = torch.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z

