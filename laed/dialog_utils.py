from __future__ import print_function

from laed import main as engine
from laed.enc2dec.decoders import GEN, DecoderRNN, TEACH_FORCE
from laed import utils
from laed.dataset.corpora import PAD, EOS, EOT, USR, SYS
import logging
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
from laed.main import LossManager
import torch


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
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
        # move from GPU to CPU
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.cpu().data.numpy()
        # get attention if possible
        if config.use_attn:
            pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0,1)
        else:
            pred_attns = None

        ctx = batch.get('contexts')
        ctx_size = ctx.shape[1]
        for b_id in range(pred_labels.shape[0]):
            pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                             b_id, attn=pred_attns)
            ctx_str = []
            for i in range(ctx_size):
                temp, _ = engine.get_sent(model, de_tknize, ctx[:, i, 1:], b_id)
                if temp:
                    ctx_str.append(temp)
            ctx_str = '<t>'.join(ctx_str)
            true_str, _ = engine.get_sent(model, de_tknize, true_labels, b_id)
            evaluator.add_example(true_str, pred_str)
            if dest_f is None:
                logger.info("Source: {}".format(ctx_str))
                logger.info("Target: {}".format(true_str))
                logger.info("Predict: {}\n".format(pred_str))
            else:
                dest_f.write("Source: {}\n".format(ctx_str))
                dest_f.write("Target: {}\n".format(true_str))
                dest_f.write("Predict: {}\n\n".format(pred_str))
    if dest_f is None:
        logging.info(evaluator.get_report(include_error=dest_f is not None))
    else:
        dest_f.write(evaluator.get_report(include_error=dest_f is not None))
    logger.info("Generation Done")


def generate_with_adv(model, data_feed, config, evaluator, num_batch=1, dest_f=None):
    eos_id = model.rev_vocab[EOS]
    model.eval()

    logger.info("Generation with Adversarial: {} batches".format(data_feed.num_batch
                                                                 if num_batch is None
                                                                 else num_batch))

    adv_nll = 0.0
    corr_cnt = 0.0
    adv_cnt = 0.0
    corr_pi_cnt = 0.0
    sys_pi_cnt = 1e-18
    sys_corr_pi_cnt = 0.0

    usr_pi_cnt = 1e-18
    usr_corr_pi_cnt = 0.0

    if num_batch is not None:
        gen_with_cond(model, data_feed, config, num_batch)

    data_feed.epoch_init(config, shuffle=False, verbose=False)
    while True:
        batch = data_feed.next_batch()
        if batch is None:
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        try:
            y_ids = outputs[DecoderRNN.KEY_LATENT]
            qy_ids = outputs[DecoderRNN.KEY_RECOG_LATENT]
            log_py = outputs[DecoderRNN.KEY_POLICY]
            _, max_py = torch.max(log_py, dim=1)
            corr_flag = max_py == qy_ids.view(-1)
            corr_flag = corr_flag.cpu().data.numpy()

            corr_pi_cnt += np.sum(corr_flag)
            for b_id in range(config.batch_size):
                for y_idx in range(config.y_size):
                    idx = b_id * config.y_size + y_idx

                    if model.rev_vocab.get(USR) in batch.outputs[b_id]:
                        usr_pi_cnt += 1
                        if corr_flag[idx]:
                            usr_corr_pi_cnt += 1
                    else:
                        sys_pi_cnt += 1
                        if corr_flag[idx]:
                            sys_corr_pi_cnt += 1

        except Exception as e:
            logger.error(e)
            logger.info("No latent. Skip")
            return

        seq_terminate = Variable(torch.zeros(config.batch_size, 1))
        if config.use_gpu:
            seq_terminate = seq_terminate.cuda().long()

        # find adversarial loss here. EOS the sequence
        norm_out_utts = []
        for t in outputs[DecoderRNN.KEY_SEQUENCE]:
            is_eos = t == eos_id
            seq_terminate = torch.sign(seq_terminate+is_eos.long())
            norm_out_utts.append((1.0-seq_terminate)*t)

        norm_out_utts = torch.cat(norm_out_utts, dim=1)
        qzx_results = model.qzx_forward(norm_out_utts)
        log_qy = F.log_softmax(qzx_results.qy_logits, dim=1)

        nll = -1.0 * log_qy.gather(1, y_ids)
        _, max_qy = torch.max(log_qy, dim=1)

        corr_cnt += torch.sum(max_qy==y_ids.view(-1)).cpu().data.numpy()
        adv_nll += torch.sum(nll).cpu().data.numpy()
        adv_cnt += log_qy.size(0)

    # print adversarial PPL
    avg_adv_nll = adv_nll/adv_cnt
    acc = corr_cnt/adv_cnt
    pi_acc = corr_pi_cnt/adv_cnt
    usr_pi_acc = usr_corr_pi_cnt/usr_pi_cnt
    sys_pi_acc = sys_corr_pi_cnt/sys_pi_cnt

    logger.info("Adversarial NLL {}, PPL {} Acc {} PI Acc {} Sys Acc {} Usr Acc {}"
                .format(avg_adv_nll, np.exp(avg_adv_nll), acc,
                        pi_acc, sys_pi_acc, usr_pi_acc))

    logger.info("Generation Done")
    return pi_acc


def dump_latent(model, data_feed, config, dest_f, num_batch=1):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, verbose=False, shuffle=False)
    logger.info("Dumping: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    all_zs = []
    all_labels = []
    all_ctxs = []
    while True:
        batch = data_feed.next_batch()
        if batch is None or (num_batch is not None
                             and data_feed.ptr > num_batch):
            break
        results = model(batch, mode=TEACH_FORCE, return_latent=True)
        labels = batch.outputs
        ctx = batch.contexts
        ctx_size = ctx.shape[1]
        log_py = results.log_py.cpu().squeeze(0).data
        log_qy = results.log_qy.cpu().squeeze(0).data
        y_ids = results.y_ids.cpu().data
        dec_init = results.dec_init_state.cpu().squeeze().data

        for b_id in range(labels.shape[0]):
            true_str, _ = engine.get_sent(model, de_tknize, labels, b_id)
            all_labels.append(true_str)
            ctx_str = []
            for i in range(ctx_size):
                temp, _ = engine.get_sent(model, de_tknize, ctx[:, i, :], b_id)
                ctx_str.append(temp)

            ctx_str = '<t>'.join(ctx_str)
            all_ctxs.append(ctx_str)

        all_zs.append((log_qy.numpy(), log_py.numpy(), y_ids.numpy(), dec_init.numpy()))

    pickle.dump({'z': all_zs, 'labels': all_labels,
                 'contexts': all_ctxs}, dest_f)
    logger.info("Dumping Done")


def gen_with_cond(model, data_feed, config, num_batch=1, dest_f=None):
    model.eval()
    old_batch_size = config.batch_size
    if num_batch != None:
        config.batch_size = 3

    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)

    logger.info("Generation: {} batches".format(data_feed.num_batch
                                                if num_batch is None
                                                else num_batch))
    print_cnt = 0
    sample_n = 5

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

        ctx = batch.get('contexts')
        ctx_size = ctx.shape[1]
        sample_outputs, _ = model(batch, mode=GEN, gen_type="sample", sample_n=sample_n)
        greedy_outputs, labels = model(batch, mode=GEN, gen_type="greedy", sample_n=sample_n)

        # move from GPU to CPU
        labels = labels.cpu()
        sample_labels = [t.cpu().data.numpy() for t in sample_outputs[DecoderRNN.KEY_SEQUENCE]]
        greedy_labels = [t.cpu().data.numpy() for t in greedy_outputs[DecoderRNN.KEY_SEQUENCE]]
        log_py = greedy_outputs[DecoderRNN.KEY_POLICY]

        greedy_y_ids = greedy_outputs[DecoderRNN.KEY_LATENT].cpu().data.numpy()
        sample_y_ids = sample_outputs[DecoderRNN.KEY_LATENT].cpu().data.numpy()

        sample_labels = np.array(sample_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        greedy_labels = np.array(greedy_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.data.numpy()

        for b_id in range(true_labels.shape[0]):
            ctx_str = []
            for i in range(ctx_size):
                temp, _ = engine.get_sent(model, de_tknize, ctx[:, i, :], b_id)
                if temp:
                    ctx_str.append(temp)
            ctx_str = '<t>'.join(ctx_str)

            true_str, _ = engine.get_sent(model, de_tknize, true_labels, b_id)
            print_cnt += 1
            write("Source: {}".format(ctx_str))
            write("Target: {}".format(true_str))
            for n_id in range(sample_n):
                pred_str, attn = engine.get_sent(model, de_tknize, greedy_labels, b_id+config.batch_size*n_id)
                code = map(str, greedy_y_ids[b_id+config.batch_size*n_id])
                write("Sample Z ({}): {}".format(" ".join(code), pred_str))
            for n_id in range(sample_n):
                pred_str, attn = engine.get_sent(model, de_tknize, sample_labels, b_id+config.batch_size*n_id)
                code = map(str, sample_y_ids[b_id+config.batch_size*n_id])
                write("Sample W ({}): {}".format(" ".join(code), pred_str))
            write('\n')
    config.batch_size = old_batch_size

    logger.info("Generation Done\n")


def selective_generate(model, data_feed, config, selected_clusters):
    model.eval()
    de_tknize = utils.get_dekenize()
    data_feed.epoch_init(config, shuffle=False, verbose=False)
    # get all code
    codes = set([d['code'] for d in selected_clusters])

    logger.info("Generation: {} batches".format(data_feed.num_batch))
    data = []
    total_cnt = 0.0
    in_cnt = 0.0

    while True:
        batch = data_feed.next_batch()
        if batch is None:
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
        # move from GPU to CPU
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        true_labels = labels.cpu().data.numpy()
        y_ids = outputs[DecoderRNN.KEY_LATENT].cpu().data.numpy()
        y_ids = y_ids.reshape(-1, config.y_size)
        ctx = batch.get('contexts')
        ctx_size = ctx.shape[1]

        for b_id in range(pred_labels.shape[0]):
            y_id = map(str, y_ids[b_id])
            code = '-'.join(y_id)
            total_cnt +=1
            if code in codes:
                pred_str, attn = engine.get_sent(model, de_tknize, pred_labels,
                                                 b_id, attn=None)
                ctx_str = []
                for i in range(ctx_size):
                    temp, _ = engine.get_sent(model, de_tknize, ctx[:, i, 1:], b_id)
                    ctx_str.append(temp)
                ctx_str = '<t>'.join(ctx_str)
                true_str, _ = engine.get_sent(model, de_tknize, true_labels, b_id)
                in_cnt +=1
                data.append({'context': ctx_str, 'target': true_str,
                             'predict': pred_str, 'code': code})

    logger.info("In rate {}".format(in_cnt/total_cnt))
    return data



