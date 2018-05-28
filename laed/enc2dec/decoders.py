# @Time    : 9/20/17 2:11 PM
# @Author  : Tiancheng Zhao

import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from laed.enc2dec.base_modules import BaseRNN
from laed.utils import FLOAT, LONG, cast_type

TEACH_FORCE = "teacher_forcing"
TEACH_GEN = "teacher_gen"
GEN = "gen"


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i - max_i x_i) / sum_j exp(x_j - max_i x_i) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dec_size, attn_size, mode, project=False):
        super(Attention, self).__init__()
        self.mask = None
        self.mode = mode
        self.attn_size = attn_size
        self.dec_size = dec_size

        if project:
            self.linear_out = nn.Linear(dec_size+attn_size, dec_size)
        else:
            self.linear_out = None

        if mode == 'general':
            self.attn_w = nn.Linear(dec_size, attn_size)
        elif mode == 'cat':
            self.dec_w = nn.Linear(dec_size, dec_size)
            self.attn_w = nn.Linear(attn_size, dec_size)
            self.query_w = nn.Linear(dec_size, 1)

    def forward(self, output, context):
        """
        :param output: [batch, out_len, dec_size]
        :param context: [batch, in_len, attn_size]
        :return: output, attn
        """
        batch_size = output.size(0)
        input_size = context.size(1)

        # batch, out_len, in_len
        if self.mode == 'dot':
            attn = torch.bmm(output, context.transpose(1, 2))
        elif self.mode == 'general':
            mapped_output = self.attn_w(output)
            attn = torch.bmm(mapped_output, context.transpose(1, 2))
        elif self.mode == 'cat':
            mapped_attn = self.attn_w(context)
            mapped_out = self.dec_w(output)
            tiled_out = mapped_out.unsqueeze(2).repeat(1, 1, input_size, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = F.tanh(tiled_attn+tiled_out)
            attn = self.query_w(fc1).squeeze(-1)

        else:
            raise ValueError("Unknown attention")

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim)
        #  -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)
        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)

        if self.linear_out is None:
            return combined, attn
        else:
            # output -> (batch, out_len, dim)
            output = F.tanh(
                self.linear_out(combined.view(-1, self.dec_size+self.attn_size))).view(
                batch_size, -1, self.dec_size)
            return output, attn


class DecoderRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, input_size, hidden_size, sos_id,
                 eos_id, n_layers=1, rnn_cell='lstm', input_dropout_p=0,
                 dropout_p=0, use_attention=False, attn_mode='cat',
                 attn_size=None, use_gpu=True, embedding=None, output_size=None,
                 tie_output_embed=False):

        super(DecoderRNN, self).__init__(vocab_size, input_size,
                                         hidden_size, input_dropout_p,
                                         dropout_p, n_layers, rnn_cell, False)

        self.output_size = vocab_size if output_size is None else output_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.use_gpu = use_gpu

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, self.input_size)
        else:
            self.embedding = embedding

        if use_attention:
            self.attention = Attention(self.hidden_size, attn_size, attn_mode,
                                       project=True)

        if tie_output_embed:
            self.project = lambda x: x * self.embedding.weight.transpose(0, 1)
        else:
            self.project = nn.Linear(self.hidden_size, self.output_size)
        self.function = F.log_softmax

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = output.contiguous()
        logits = self.project(output.view(-1, self.hidden_size))
        predicted_softmax = self.function(logits, dim=logits.dim()-1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, batch_size, inputs=None, init_state=None,
                attn_context=None, mode=TEACH_FORCE, gen_type='greedy',
                beam_size=4):

        # sanity checks
        ret_dict = dict()

        if self.use_attention:
            # calculate initial attention
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if mode == GEN:
            inputs = None

        if gen_type != 'beam':
            beam_size = 1

        if inputs is not None:
            decoder_input = inputs
        else:
            # prepare the BOS inputs
            bos_var = Variable(torch.LongTensor([self.sos_id]), volatile=True)
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size*beam_size, 1)

        if mode == GEN and gen_type == 'beam':
            # if beam search, repeat the initial states of the RNN
            if self.rnn_cell is nn.LSTM:
                h, c = init_state
                decoder_hidden = (self.repeat_state(h, batch_size, beam_size),
                                  self.repeat_state(c, batch_size, beam_size))
            else:
                decoder_hidden = self.repeat_state(init_state,
                                                   batch_size, beam_size)
        else:
            decoder_hidden = init_state

        decoder_outputs = [] # a list of logprob
        sequence_symbols = [] # a list word ids
        back_pointers = [] # a list of parent beam ID
        lengths = np.array([self.max_length] * batch_size * beam_size)

        def decode(step, cum_sum, step_output, step_attn):
            decoder_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)

            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            elif gen_type == 'beam':
                if step == 0:
                    seq_score = step_output_slice.view(batch_size, -1)
                    seq_score = seq_score[:, 0:self.output_size]
                else:
                    seq_score = cum_sum + step_output_slice
                    seq_score = seq_score.view(batch_size, -1)

                top_v, top_id = seq_score.topk(beam_size)

                back_ptr = top_id.div(self.output_size).view(-1, 1)
                symbols = top_id.fmod(self.output_size).view(-1, 1)
                cum_sum = top_v.view(-1, 1)
                back_pointers.append(back_ptr)
            else:
                raise ValueError("Unsupported decoding mode")

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return cum_sum, symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph
        if mode == TEACH_FORCE:
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, attn_context)

            # in teach forcing mode, we don't need symbols.
            decoder_outputs = decoder_output

        else:
            # do free running here
            cum_sum = None
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(
                    decoder_input, decoder_hidden, attn_context)

                cum_sum, symbols = decode(di, cum_sum, decoder_output, step_attn)
                decoder_input = symbols

            decoder_outputs = torch.cat(decoder_outputs, dim=1)

            if gen_type == 'beam':
                # do back tracking here to recover the 1-best according to
                # beam search.
                final_seq_symbols = []
                cum_sum = cum_sum.view(-1, beam_size)
                max_seq_id = cum_sum.topk(1)[1].data.cpu().view(-1).numpy()
                rev_seq_symbols = sequence_symbols[::-1]
                rev_back_ptrs = back_pointers[::-1]

                for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
                    symbol2ds = symbols.view(-1, beam_size)
                    back2ds = back_ptrs.view(-1, beam_size)

                    selected_symbols = []
                    selected_parents =[]
                    for b_id in range(batch_size):
                        selected_parents.append(back2ds[b_id, max_seq_id[b_id]])
                        selected_symbols.append(symbol2ds[b_id, max_seq_id[b_id]])

                    final_seq_symbols.append(torch.cat(selected_symbols).unsqueeze(1))
                    max_seq_id = torch.cat(selected_parents).data.cpu().numpy()
                sequence_symbols = final_seq_symbols[::-1]

        # save the decoded sequence symbols and sequence length
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict




