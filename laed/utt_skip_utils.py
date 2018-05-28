# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import print_function


from laed import main as engine
from laed.enc2dec.decoders import GEN, DecoderRNN, TEACH_FORCE
from laed import utils
import torch
import logging
import numpy as np
import pickle
from collections import defaultdict


logger = logging.getLogger()


def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size

    if num_batch != None:
        config.batch_size = 5

    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

    evaluator.initialize()
    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))

    def write(msg):
        if dest_f is None:
            logger.info(msg)
        else:
            dest_f.write(msg+'\n')

    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
        prev_outputs, next_outputs = outputs
        prev_labels, next_labels = labels
        cur_labels = batch.get('outputs')

        prev_labels = prev_labels.cpu().data.numpy()
        next_labels = next_labels.cpu().data.numpy()

        prev_pred = [t.cpu().data.numpy() for t in prev_outputs[DecoderRNN.KEY_SEQUENCE]]
        prev_pred = np.array(prev_pred, dtype=int).squeeze(-1).swapaxes(0,1)

        next_pred = [t.cpu().data.numpy() for t in next_outputs[DecoderRNN.KEY_SEQUENCE]]
        next_pred = np.array(next_pred, dtype=int).squeeze(-1).swapaxes(0,1)

        for b_id in range(cur_labels.shape[0]):
            ctx_str, _ = engine.get_sent(model, de_tknize, cur_labels, b_id)
            prev_true_str, _ = engine.get_sent(model, de_tknize, prev_labels,
                                               b_id)
            next_true_str, _ = engine.get_sent(model, de_tknize, next_labels,
                                               b_id)

            pred_prev_str, _ = engine.get_sent(model, de_tknize, prev_pred, b_id)
            pred_next_str, _ = engine.get_sent(model, de_tknize, next_pred, b_id)

            evaluator.add_example(prev_true_str, pred_prev_str)
            evaluator.add_example(next_true_str, pred_next_str)

            write("Response: {}".format(ctx_str))
            write("Prev Target: {}".format(prev_true_str))
            write("Prev Predict: {}".format(pred_prev_str))
            write("Next Target: {}".format(next_true_str))
            write("Next Predict: {}\n".format(pred_next_str))

    if dest_f is None:
        logging.info(evaluator.get_report(include_error=dest_f is not None))
    else:
        dest_f.write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")


def dump_latent(model, data_feed, config, dest_f, num_batch=1):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Dumping: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    all_zs = []
    all_labels = []
    all_metas = []
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas
        log_qy = results.log_qy.cpu().squeeze(0).data
        y_ids = results.y_ids.cpu().data
        dec_init = results.dec_init_state.cpu().squeeze().data

        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            all_labels.append(true_str)
            all_metas.append(metas[b_id])

        all_zs.append((log_qy.numpy(), dec_init.numpy(), y_ids.numpy()))

    pickle.dump({'z': all_zs, 'labels': all_labels, "metas": all_metas}, dest_f)
    logger.info("Dumping Done")

