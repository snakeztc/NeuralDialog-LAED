# @Time    : 9/25/17 3:54 PM
# @Author  : Tiancheng Zhao
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from laed.utils import get_dekenize, get_tokenize
from scipy.stats import gmean
import logging
from laed.dataset.corpora import EOS, BOS
from collections import defaultdict


class EvaluatorBase(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp, domain='default'):
        raise NotImplementedError

    def get_report(self, include_error=False):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BleuEvaluator(EvaluatorBase):
    """
    Use string matching to find the F-1 score of slots
    Use logistic regression to find F-1 score of acts
    Use string matching to find F-1 score of KB_SEARCH
    """
    logger = logging.getLogger(__name__)

    def __init__(self, data_name):
        self.data_name = data_name
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def initialize(self):
        self.domain_labels = defaultdict(list)
        self.domain_hyps = defaultdict(list)

    def add_example(self, ref, hyp, domain='default'):
        self.domain_labels[domain].append(ref)
        self.domain_hyps[domain].append(hyp)

    def get_report(self, include_error=False):
        reports = []
        tokenize = get_tokenize()

        for domain, labels in self.domain_labels.items():
            predictions = self.domain_hyps[domain]
            self.logger.info("Generate report for {} for {} samples".format(domain, len(predictions)))
            refs, hyps = [], []
            for label, hyp in zip(labels, predictions):
                label = label.replace(EOS, '').replace(BOS, '')
                hyp = hyp.replace(EOS, '').replace(BOS, '')
                ref_tokens = tokenize(label)[2:]
                hyp_tokens = tokenize(hyp)[2:]

                refs.append([ref_tokens])
                hyps.append(hyp_tokens)

            # compute corpus level scores
            bleu = bleu_score.corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
            report = "\nDomain: %s BLEU %f\n" % (domain, bleu)
            reports.append(report)

        return "\n==== REPORT===={report}".format(report="========".join(reports))




