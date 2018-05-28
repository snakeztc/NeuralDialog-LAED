# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
import torch
import torch.nn as nn
import torch.nn.functional as F
from laed.dataset.corpora import PAD, BOS, EOS
from torch.autograd import Variable
from laed import criterions
from laed.enc2dec.decoders import DecoderRNN
from laed.enc2dec.encoders import EncoderRNN
from laed.utils import INT, FLOAT, LONG, cast_type
from laed import nn_lib
import numpy as np
from laed.models.model_bases import BaseModel
from laed.enc2dec.decoders import GEN
from laed.utils import Pack
import itertools


class DiVAE(BaseModel):
    def __init__(self, corpus, config):
        super(DiVAE, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size * 2 if self.bi_enc_cell else self.enc_cell_size

        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=self.config.fix_batch,
                                    bidirection=self.bi_enc_cell)

        self.q_y = nn.Linear(self.enc_out_size, config.y_size * config.k)
        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.y_size * config.k,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm',
                                                         has_bias=False)

        self.decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                  self.embed_size, self.dec_cell_size,
                                  self.go_id, self.eos_id,
                                  n_layers=1, rnn_cell=self.rnn_cell,
                                  input_dropout_p=self.dropout,
                                  dropout_p=self.dropout,
                                  use_attention=self.use_attn,
                                  attn_size=self.enc_cell_size,
                                  attn_mode=self.attn_type,
                                  use_gpu=self.use_gpu,
                                  embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        self.log_py = nn.Parameter(torch.log(torch.ones(self.config.y_size,
                                                        self.config.k)/config.k),
                                   requires_grad=True)
        self.register_parameter('log_py', self.log_py)

        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()

        self.kl_w = 0.0

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.nll
        if self.config.use_reg_kl:
            total_loss += loss.reg_kl

        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        batch_size = len(data_feed['output_lens'])
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.transpose(0, 1).contiguous().view(-1,
                                                              self.enc_out_size)

        # x_last = torch.mean(x_outs, dim=1)

        # posterior network
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, qy_logits.dim()-1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qy_logits, 1.0, self.use_gpu, hard=not self.training, return_max_id=True)
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        y_ids = y_ids.view(-1, self.config.y_size)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   mode=mode, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        if mode == GEN:
            return dec_ctx, labels
        else:
            # RNN reconstruction
            nll = self.nll_loss(dec_outs, labels)

            # regularization qy to be uniform
            avg_log_qy = torch.exp(log_qy.view(-1, self.config.y_size, self.config.k))
            avg_log_qy = torch.log(torch.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y,
                                          batch_size, unit_average=True)

            if self.config.use_mutual:
                if self.config.train_prior:
                    reg_kl = self.cat_kl_loss(avg_log_qy, F.log_softmax(self.log_py, self.log_py.dim()-1),
                                              batch_size, unit_average=True)
                else:
                    reg_kl = b_pr
            else:
                reg_kl = self.cat_kl_loss(log_qy, self.log_uniform_y,
                                          batch_size, unit_average=True)

            # find out mutual information
            # H(Z) - H(Z|X)
            mi = self.entropy_loss(avg_log_qy, unit_average=True)\
                 - self.entropy_loss(log_qy, unit_average=True)

            results = Pack(nll=nll, reg_kl=reg_kl, mi=mi, bpr=b_pr)

            if return_latent:
                results['log_qy'] = log_qy
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids

            return results

    def sweep(self, data_feed, gen_type='greedy'):
        ctx_lens = data_feed['output_lens']
        batch_size = len(ctx_lens)
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)

        # posterior network
        qy_logits = self.q_y(x_last).view(-1, self.config.k)

        # switch that controls the sampling
        sample_y, y_id = self.cat_connector(qy_logits, 1.0, self.use_gpu,
                                            hard=True, return_max_id=True)
        y_id = y_id.view(-1, self.config.y_size)
        start_y_id = y_id[0]
        end_y_id = y_id[batch_size-1]

        # start sweeping
        all_y_ids = [start_y_id]
        for idx in range(self.config.y_size):
            mask = torch.zeros(self.config.y_size)
            mask[0:idx+1] = 1.0
            neg_mask = 1 - mask
            mask = cast_type(Variable(mask), LONG, self.use_gpu)
            neg_mask = cast_type(Variable(neg_mask), LONG, self.use_gpu)
            temp_y = neg_mask * start_y_id + mask * end_y_id
            all_y_ids.append(temp_y)
        num_steps = len(all_y_ids)
        all_y_ids = torch.cat(all_y_ids, dim=0).view(num_steps, -1)

        sample_y = cast_type(Variable(torch.zeros((num_steps*self.config.y_size, self.config.k))), FLOAT, self.use_gpu)
        sample_y.scatter_(1, all_y_ids.view(-1, 1), 1.0)
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        batch_size = num_steps

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        labels = out_utts[:, 1:].contiguous()
        dec_inputs = out_utts[:, 0:-1]

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   dec_inputs, dec_init_state,
                                                   mode=GEN, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        return dec_ctx, labels, all_y_ids

    def enumerate(self, repeat=1, gen_type='greedy'):

        # do something here. For each y, we enumerate from 0 to K
        # and take the expectation of other values.
        batch_size = self.config.y_size * self.config.k * repeat
        sample_y = cast_type(Variable(torch.zeros((batch_size,
                                                   self.config.y_size,
                                                   self.config.k))),
                             FLOAT, self.use_gpu)
        sample_y += 1.0/self.config.k

        for y_id in range(self.config.y_size):
            for k_id in range(self.config.k):
                for r_id in range(repeat):
                    idx = y_id*self.config.k + k_id*repeat + r_id
                    sample_y[idx, y_id] = 0.0
                    sample_y[idx, y_id, k_id] = 1.0

        # map sample to initial state of decoder
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.dec_init_connector(sample_y)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   None, dec_init_state,
                                                   mode=GEN, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        # compute loss or return results
        return dec_ctx

    def exp_enumerate(self, repeat=1, gen_type='greedy'):

        # do something here. For each y, we enumerate from 0 to K
        # and take the expectation of other values.
        batch_size = np.power(self.config.k, self.config.y_size) * repeat
        sample_y = cast_type(Variable(torch.zeros((batch_size*self.config.y_size,
                                                   self.config.k))),
                             FLOAT, self.use_gpu)
        d = dict((str(i), range(self.config.k)) for i in range(self.config.y_size))
        all_y_ids = []
        for combo in itertools.product(*[d[k] for k in sorted(d.keys())]):
            all_y_ids.append(list(combo))
        np_y_ids = np.array(all_y_ids)
        np_y_ids = self.np2var(np_y_ids, LONG)
        # map sample to initial state of decoder
        sample_y.scatter_(1, np_y_ids.view(-1, 1), 1.0)
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        dec_init_state = self.dec_init_connector(sample_y)

        # decode
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size,
                                                   None, dec_init_state,
                                                   mode=GEN, gen_type=gen_type,
                                                   beam_size=self.beam_size)
        return dec_ctx, all_y_ids



class DiVST(BaseModel):
    def __init__(self, corpus, config):
        super(DiVST, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.embed_size = config.embed_size
        self.max_utt_len = config.max_utt_len
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.num_layer = config.num_layer
        self.dropout = config.dropout
        self.enc_cell_size = config.enc_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.rnn_cell = config.rnn_cell
        self.max_dec_len = config.max_dec_len
        self.use_attn = config.use_attn
        self.beam_size = config.beam_size
        self.utt_type = config.utt_type
        self.bi_enc_cell = config.bi_enc_cell
        self.attn_type = config.attn_type
        self.enc_out_size = self.enc_cell_size*2 if self.bi_enc_cell else self.enc_cell_size


        # build model here
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.x_encoder = EncoderRNN(self.embed_size, self.enc_cell_size,
                                    bidirection=self.bi_enc_cell,
                                    dropout_p=self.dropout,
                                    rnn_cell=self.rnn_cell,
                                    variable_lengths=False)

        self.q_y = nn.Linear(self.enc_out_size, config.y_size * config.k)
        self.cat_connector = nn_lib.GumbelConnector()
        self.dec_init_connector = nn_lib.LinearConnector(config.y_size * config.k,
                                                         self.dec_cell_size,
                                                         self.rnn_cell == 'lstm')


        self.prev_decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                       self.embed_size, self.dec_cell_size,
                                       self.go_id, self.eos_id,
                                       n_layers=1, rnn_cell=self.rnn_cell,
                                       input_dropout_p=self.dropout,
                                       dropout_p=self.dropout,
                                       use_attention=self.use_attn,
                                       attn_size=self.enc_cell_size,
                                       attn_mode=self.attn_type,
                                       use_gpu=self.use_gpu,
                                       embedding=self.embedding)

        self.next_decoder = DecoderRNN(self.vocab_size, self.max_dec_len,
                                       self.embed_size, self.dec_cell_size,
                                       self.go_id, self.eos_id,
                                       n_layers=1, rnn_cell=self.rnn_cell,
                                       input_dropout_p=self.dropout,
                                       dropout_p=self.dropout,
                                       use_attention=self.use_attn,
                                       attn_size=self.enc_cell_size,
                                       attn_mode=self.attn_type,
                                       use_gpu=self.use_gpu,
                                       embedding=self.embedding)

        self.nll_loss = criterions.NLLEntropy(self.rev_vocab[PAD], self.config)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.cross_ent_loss = criterions.CrossEntropyoss()
        self.entropy_loss = criterions.Entropy()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
        self.kl_w = 1.0

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = loss.prev_nll + loss.next_nll
        if self.config.use_reg_kl:
            total_loss += loss.reg_kl

        return total_loss

    def forward(self, data_feed, mode, gen_type='greedy', sample_n=1, return_latent=False):
        ctx_lens = data_feed['output_lens']
        batch_size = len(ctx_lens)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        prev_utts = self.np2var(data_feed['prevs'], LONG)
        next_utts = self.np2var(data_feed['nexts'], LONG)

        # output encoder
        output_embedding = self.embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        if type(x_last) is tuple:
            x_last = x_last[0].transpose(0, 1).contiguous().view(-1, self.enc_out_size)
        else:
            x_last = x_last.transpose(0, 1).contiguous().view(-1, self.enc_out_size)

        # posterior network
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, dim=qy_logits.dim()-1)

        # switch that controls the sampling
        sample_y, y_ids = self.cat_connector(qy_logits, self.kl_w, self.use_gpu, hard=not self.training, return_max_id=True)
        sample_y = sample_y.view(-1, self.config.k * self.config.y_size)
        y_ids = y_ids.view(-1, self.config.y_size)

        # map sample to initial state of decoder
        dec_init_state = self.dec_init_connector(sample_y)

        # get decoder inputs
        prev_labels = prev_utts[:, 1:].contiguous()
        next_labels = next_utts[:, 1:].contiguous()
        prev_dec_inputs = prev_utts[:, 0:-1]
        next_dec_inputs = next_utts[:, 0:-1]

        # decode
        prev_dec_outs, prev_dec_last, prev_dec_ctx = self.prev_decoder(batch_size,
                                                                       prev_dec_inputs, dec_init_state,
                                                                       mode=mode, gen_type=gen_type,
                                                                       beam_size=self.beam_size)

        next_dec_outs, next_dec_last, next_dec_ctx = self.next_decoder(batch_size,
                                                                       next_dec_inputs, dec_init_state,
                                                                       mode=mode, gen_type=gen_type,
                                                                       beam_size=self.beam_size)

        # compute loss or return results
        if mode == GEN:
            return (prev_dec_ctx, next_dec_ctx), (prev_labels, next_labels)
        else:
            # RNN reconstruction
            prev_nll = self.nll_loss(prev_dec_outs, prev_labels)
            next_nll = self.nll_loss(next_dec_outs, next_labels)

            # regularization qy to be uniform
            avg_log_qy = torch.exp(log_qy.view(-1, self.config.y_size,
                                               self.config.k))
            avg_log_qy = torch.log(torch.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y,
                                    batch_size, unit_average=True)

            if self.config.use_mutual:
                reg_kl = b_pr
            else:
                reg_kl = self.cat_kl_loss(log_qy, self.log_uniform_y,
                                          batch_size, unit_average=True)

            # find out mutual information
            # H(Z) - H(Z|X)
            h_qz = self.entropy_loss(avg_log_qy, unit_average=True)
            h_qzx = self.entropy_loss(log_qy, unit_average=True)
            mi = h_qz - h_qzx

            results = Pack(prev_nll=prev_nll, next_nll=next_nll, reg_kl=reg_kl,
                           mi=mi, b_pr=b_pr)

            if return_latent:
                results['log_qy'] = log_qy
                results['dec_init_state'] = dec_init_state
                results['y_ids'] = y_ids

            return results


