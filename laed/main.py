# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import print_function
import numpy as np
from laed.models.model_bases import summary
import torch
from laed.dataset.corpora import PAD, EOS, EOT
from laed.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from laed.utils import get_dekenize
import os
from collections import defaultdict
import logging

logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.data[0])

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.data[0])

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            avg_loss = np.average(loss) if window is None else np.average(loss[-window:])
            str_losses.append("{} {:.3f}".format(key, avg_loss))
            if 'nll' in key:
                str_losses.append("PPL {:.3f}".format(np.exp(avg_loss)))
        if prefix:
            return "{}: {} {}".format(prefix, name, " ".join(str_losses))
        else:
            return "{} {}".format(name, " ".join(str_losses))

    def avg_loss(self):
        return np.mean(self.backward_losses)


def get_sent(model, de_tknize, data, b_id, attn=None, attn_ctx=None, stop_eos=True, stop_pad=True):
    ws = []
    attn_ws = []
    has_attn = attn is not None and attn_ctx is not None
    for t_id in range(data.shape[1]):
        w = model.vocab[data[b_id, t_id]]
        if has_attn:
            a_val = np.max(attn[b_id, t_id])
            if a_val > 0.1:
                a = np.argmax(attn[b_id, t_id])
                attn_w = model.vocab[attn_ctx[b_id, a]]
                attn_ws.append("{}({})".format(attn_w, a_val))
        if (stop_eos and w in [EOS, EOT]) or (stop_pad and w == PAD):
            if w == EOT:
                ws.append(w)
            break
        if w != PAD:
            ws.append(w)

    att_ws = "Attention: {}".format(" ".join(attn_ws)) if attn_ws else ""
    if has_attn:
        return de_tknize(ws), att_ws
    else:
        try:
            return de_tknize(ws), ""
        except:
            return " ".join(ws), ""


def train(model, train_feed, valid_feed, test_feed, config, evaluator, gen=None):
    if gen is None:
        gen = generate

    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    train_loss = LossManager()
    model.train()
    logger.info(summary(model, show_weights=False))
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))

    while True:
        train_feed.epoch_init(config, verbose=done_epoch==0, shuffle=True)
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break

            optimizer.zero_grad()
            loss = model(batch, mode=TEACH_FORCE)
            if model.flush_valid:
                logger.info("Flush previous valid loss")
                best_valid_loss = np.inf
                model.flush_valid = False
                optimizer = model.get_optimizer(config)

            model.backward(batch_cnt, loss)
            optimizer.step()
            batch_cnt += 1
            train_loss.add_loss(loss)

            if batch_cnt % config.print_step == 0:
                logger.info(train_loss.pprint("Train", window=config.print_step,
                                              prefix="{}/{}-({:.3f})".format(batch_cnt % config.ckpt_step,
                                                                         config.ckpt_step,
                                                                         model.kl_w)))

            if batch_cnt % config.ckpt_step == 0:
                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                done_epoch += 1

                # validation
                valid_loss = validate(model,valid_feed, config, batch_cnt)

                # generating
                gen(model, test_feed, config, evaluator, num_batch=config.preview_batch_num)

                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience,
                                       done_epoch * config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info("Update patience to {}".format(patience))

                    if config.save_model:
                        logger.info("Model Saved.")
                        torch.save(model.state_dict(),
                                   os.path.join(config.session_dir, "model"))

                    best_valid_loss = valid_loss

                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch:
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")

                    logger.info("Best validation loss %f" % best_valid_loss)

                    return

                # exit eval model
                model.train()
                train_loss.clear()
                logger.info("\n**** Epcoch {}/{} ****".format(done_epoch,
                                                       config.max_epoch))


def validate(model, valid_feed, config, batch_cnt=None):
    model.eval()
    valid_feed.epoch_init(config, shuffle=False, verbose=True)
    losses = LossManager()
    while True:
        batch = valid_feed.next_batch()
        if batch is None:
            break
        loss = model(batch, mode=TEACH_FORCE)
        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()
    logger.info(losses.pprint(valid_feed.name))
    logger.info("Total valid loss {}".format(valid_loss))

    return valid_loss


def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    de_tknize = get_dekenize()

    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            logger.info(msg)
        else:
            dest_f.write(msg + '\n')

    data_feed.epoch_init(config, shuffle=num_batch is not None, verbose=False)
    evaluator.initialize()
    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in
                       outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        # get attention if possible
        if config.use_attn or config.use_ptr:
            pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0,1)
        else:
            pred_attns = None

        # get last 1 context
        ctx = batch.get('contexts')
        ctx_len = batch.get('context_lens')
        domains = batch.domains

        # logger.info the batch in String.
        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = get_sent(model, de_tknize, pred_labels, b_id, attn=pred_attns)
            true_str, _ = get_sent(model, de_tknize, true_labels, b_id)
            prev_ctx = ""
            if ctx is not None:
                ctx_str, _ = get_sent(model, de_tknize, ctx[:, ctx_len[b_id]-1, :], b_id)
                prev_ctx = "Source: {}".format(ctx_str)

            domain = domains[b_id]
            evaluator.add_example(true_str, pred_str, domain)
            if num_batch is None or num_batch <= 2:
                write(prev_ctx)
                write("{}:: True: {} ||| Pred: {}".format(domain, true_str, pred_str))
                if attn:
                    write("[[{}]]".format(attn))

    write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")


