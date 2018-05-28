from __future__ import print_function

from laed import main as engine
from laed.enc2dec.decoders import GEN, DecoderRNN, TEACH_FORCE
from laed import utils
from collections import defaultdict, Counter
import logging
import numpy as np
import pickle
import torch
from sklearn import metrics

logger = logging.getLogger()


def generate(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size

    if num_batch != None:
        config.batch_size = 3

    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

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
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()
        # get attention if possible
        pred_attns = None

        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                             b_id, attn=pred_attns)
            true_str, _ = engine.get_sent(model, de_tknize, true_labels, b_id)
            evaluator.add_example(true_str, pred_str)
            if dest_f is None:
                logger.info("Target: {}".format(true_str))
                logger.info("Predict: {}\n".format(pred_str))
            else:
                dest_f.write("Target: {}\n".format(true_str))
                dest_f.write("Predict: {}\n\n".format(pred_str))

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


def sweep(model, data_feed, config, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size

    if num_batch != None:
        config.batch_size = 10

    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    config.batch_size = old_batch_size

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
        outputs, labels, all_y_ids = model.sweep(batch, gen_type=config.gen_type)
        # move from GPU to CPU
        true_labels = labels.cpu().data.numpy()
        all_y_ids = all_y_ids.cpu().data.numpy()

        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        # get attention if possible
        pred_attns = None

        true_str, _ = engine.get_sent(model, de_tknize, true_labels, 0)
        write("Start: {}".format(true_str))
        prev_code = None
        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                             b_id, attn=pred_attns)
            code = '-'.join(map(str, all_y_ids[b_id]))
            if prev_code != code:
                write("Predict ({}): {}".format(code, pred_str))
                prev_code = code

        true_str, _ = engine.get_sent(model, de_tknize, true_labels, true_labels.shape[0]-1)
        write("End: {}\n".format(true_str))

    logger.info("Generation Done")


def latent_cluster(model, data_feed, config, num_batch=1, max_samples=5):
    if np.power(config.k, config.y_size) > 1000:
        logger.info("Skip latent cluster too many states")
        return
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Find cluster for: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))

    all_clusters = defaultdict(list)
    cond_y_matrix = np.zeros((config.k, config.k))

    def write(msg):
        logger.info(msg)

    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas
        log_qy = results.log_qy.view(-1, config.y_size, config.k)
        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()

        y_ids = results.y_ids.cpu().data.numpy()
        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            cond_y_matrix[y_ids[b_id]] += 1
            code = []
            for y_id in range(config.y_size):
                for k_id in range(config.k):
                    if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
                        code.append(str(k_id))
                        break
            code = '-'.join(code)
            all_clusters[code].append((true_str, metas[b_id]))

    # show clusters
    keys = all_clusters.keys()
    keys = sorted(keys)
    logger.info("Find {} clusters".format(len(keys)))

    selected_clusters = []
    for symbol in keys:
        sents = all_clusters[symbol]
        if len(sents) < 5:
            write("Skip tiny cluster with {} utts - {}".format(len(sents), symbol))
            continue
        write("Symbol {}".format(symbol))
        if len(sents) < max_samples:
            print("Find small cluster with {} utts".format(len(sents)))
            subset_ids = range(len(sents))
            np.random.shuffle(subset_ids)
        else:
            subset_ids = np.random.choice(range(len(sents)), max_samples, replace=False)
        for s_id in subset_ids[0:5]:
            write(sents[s_id][0])
        write("")
        selected_clusters.append({'code': symbol, 'meaning': '',
                                  'examples': [sents[idx][0] for idx in subset_ids]})


    return selected_clusters


def find_mi(model, data_feed, config):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Find MI for: {} batches".format(data_feed.num_batch))

    all_codes = []
    all_metas = []
    meta_keys = set()
    def write(msg):
        logger.info(msg)

    def code2id(code, base):
        idx = 0
        for c_id, c in enumerate(code):
            idx += int(c) * np.power(base, c_id)
        return idx

    while True:
        batch = data_feed.next_batch()
        if batch is None:
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)

        labels = batch.outputs
        metas = batch.metas
        for key in metas[0].keys():
            meta_keys.add(key)
        log_qy = results.log_qy.view(-1, config.y_size, config.k)
        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()
        y_ids = results.y_ids.cpu().data.numpy()
        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            code = []
            for y_id in range(config.y_size):
                for k_id in range(config.k):
                    if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
                        code.append(str(k_id))
                        break
            #all_codes.append(code)
            all_codes.append(y_ids[b_id])
            all_metas.append(metas[b_id])

    vec_codes = np.array(all_codes).transpose(0, 1)
    vec_idxes = [code2id(c, config.k) for c in vec_codes]
    vec_vocabs = list(set(vec_idxes))
    vec_idxes = [vec_vocabs.index(v) for v in vec_idxes]

    for key in meta_keys:
        # get all meta about this key
        meta_vals = []
        for m in all_metas:
            if type(m[key]) is list:
                meta_vals.append(" ".join(map(str, m[key])))
            elif type(m[key]) is dict:
                break
            else:
                meta_vals.append(m[key])
        if not meta_vals:
            continue
        meta_vocab = list(set(meta_vals))
        meta_vals = [meta_vocab.index(v) for v in meta_vals]

        mi = metrics.homogeneity_score(meta_vals, vec_idxes)
        write("{} mi with ID is {}".format(key, mi))

        # individual dimension
        for y_id in range(config.y_size):
            mi = metrics.homogeneity_score(meta_vals, vec_codes[:, y_id])
            write("{} mi with dim {} is {}".format(key, y_id, mi))



