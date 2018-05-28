# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from laed.utils import get_tokenize, get_chat_tokenize, missingdict, Pack
import logging
import os
import itertools
from collections import defaultdict
import copy

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
WILD = "%s"


class StanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)

        return self._process_dialog(data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'assistant': SYS, 'driver': USR}
        for raw_dialog in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]
            for turn in raw_dialog['dialogue']:
                utt = turn['data']['utterance']
                utt = [BOS, speaker_map[turn['turn']]] + self.tokenize(utt) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=turn['turn']))

            if hasattr(self.config, 'include_eod') and self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]
        print("<d> index %d" % self.rev_vocab[BOD])

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class PTBCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'ptb.train.txt'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'ptb.valid.txt'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'ptb.test.txt'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'rb') as f:
            lines = f.readlines()

        return self._process_data(lines)

    def _process_data(self, data):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + self.tokenize(line)[1:] + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for turn in self.train_corpus:
            all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=self._sent2id(line.utt),
                           speaker=line.speaker,
                           meta=line.get('meta'))
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class DailyDialogCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'train'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'validation'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'test'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(os.path.join(path, 'dialogues.txt'), 'rb') as f:
            txt_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_act.txt'), 'rb') as f:
            da_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_emotion.txt'), 'rb') as f:
            emotion_lines = f.readlines()

        combined_data = [(t, d, e) for t, d, e in zip(txt_lines, da_lines, emotion_lines)]

        return self._process_dialog(combined_data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        for raw_dialog, raw_act, raw_emotion in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]

            raw_dialog = raw_dialog.decode('ascii', 'ignore').encode()
            raw_dialog = raw_dialog.split('__eou__')[0:-1]
            raw_act = raw_act.split()
            raw_emotion = raw_emotion.split()

            for t_id, turn in enumerate(raw_dialog):
                utt = turn
                utt = [BOS] + self.tokenize(utt.lower()) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=t_id%2,
                                   meta={'emotion': raw_emotion[t_id], 'act': raw_act[t_id]}))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)
