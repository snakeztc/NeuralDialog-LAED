# @Time    : 9/20/17 2:10 PM
# @Author  : Tiancheng Zhao
import torch.nn as nn
from laed.dataset.corpora import PAD, EOS
import torch
import torch
from torch.autograd import Variable
from laed.utils import FLOAT, cast_type


class BaseRNN(nn.Module):
    SYM_MASK = PAD
    SYM_EOS = EOS

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_LATENT = 'latent'
    KEY_RECOG_LATENT = 'recog_latent'
    KEY_POLICY = "policy"
    KEY_G = 'g'
    KEY_PTR_SOFTMAX = 'ptr_softmax'
    KEY_PTR_CTX = "ptr_context"


    def __init__(self, vocab_size, input_size, hidden_size, input_dropout_p,
                 dropout_p, n_layers, rnn_cell, bidirectional):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p,
                                 bidirectional=bidirectional)
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)

    def gumbel_max(self, log_probs):
        """
        Obtain a sample from the Gumbel max. Not this is not differentibale.
        :param log_probs: [batch_size x vocab_size]
        :return: [batch_size x 1] selected token IDs
        """
        sample = torch.Tensor(log_probs.size()).uniform_(0, 1)
        sample = cast_type(Variable(sample), FLOAT, self.use_gpu)

        # compute the gumbel sample
        matrix_u = -1.0 * torch.log(-1.0 * torch.log(sample))
        gumbel_log_probs = log_probs + matrix_u
        max_val, max_ids = torch.max(gumbel_log_probs, dim=-1, keepdim=True)
        return max_ids

    def repeat_state(self, state, batch_size, times):
        new_s = state.repeat(1, 1, times)
        return new_s.view(-1, batch_size * times, self.hidden_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()