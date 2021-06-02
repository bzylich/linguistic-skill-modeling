"""
Copyright (c) 2016 Duolingo Inc. MIT Licence.

Python script that implements spaced repetition models from Settles & Meeder (2016).
Recommended to run with pypy for efficiency. See README.
"""

import argparse
import csv
import gzip
import math
import os
import random
import sys
import json
import numpy as np

from collections import defaultdict, namedtuple

from nltk.metrics.distance import edit_distance

import sentencepiece as spm
import glob


# various constraints on parameters and outputs
MIN_HALF_LIFE = 15.0 / (24 * 60)    # 15 minutes
MAX_HALF_LIFE = 274.                # 9 months
LN2 = math.log(2.)


# data instance object
Instance = namedtuple('Instance', 'p t fv h a lang right wrong ts uid lexeme'.split())


class SpacedRepetitionModel(object):
    """
    Spaced repetition model. Implements the following approaches:
      - 'hlr' (half-life regression; trainable)
      - 'lr' (logistic regression; trainable)
      - 'leitner' (fixed)
      - 'pimsleur' (fixed)
    """
    def __init__(self, method='hlr', omit_h_term=False, initial_weights=None, lrate=.001, hlwt=.01, l2wt=.1, sigma=1.):
        self.method = method
        self.omit_h_term = omit_h_term
        self.weights = defaultdict(float)
        if initial_weights is not None:
            self.weights.update(initial_weights)
        self.fcounts = defaultdict(int)
        self.lrate = lrate
        self.hlwt = hlwt
        self.l2wt = l2wt
        self.sigma = sigma

    def halflife(self, inst, base):
        try:
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            return hclip(base ** dp)
        except:
            return MAX_HALF_LIFE

    def predict(self, inst, base=2.):
        if self.method == 'hlr':
            h = self.halflife(inst, base)
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'leitner':
            try:
                h = hclip(2. ** inst.fv[0][1])
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'pimsleur':
            try:
                h = hclip(2. ** (2.35*inst.fv[0][1] - 16.46))
            except OverflowError:
                h = MAX_HALF_LIFE
            p = 2. ** (-inst.t/h)
            return pclip(p), h
        elif self.method == 'lr':
            dp = sum([self.weights[k]*x_k for (k, x_k) in inst.fv])
            p = 1./(1+math.exp(-dp))
            return pclip(p), random.random()
        else:
            raise Exception

    def train_update(self, inst):
        if self.method == 'hlr':
            base = 2.
            p, h = self.predict(inst, base)
            dlp_dw = 2.*(p-inst.p)*(LN2**2)*p*(inst.t/h)
            dlh_dw = 2.*(h-inst.h)*LN2*h
            for (k, x_k) in inst.fv:
                rate = (1./(1+inst.p)) * self.lrate / math.sqrt(1 + self.fcounts[k])
                # rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # sl(p) update
                self.weights[k] -= rate * dlp_dw * x_k
                # sl(h) update
                if not self.omit_h_term:
                    self.weights[k] -= rate * self.hlwt * dlh_dw * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.fcounts[k] += 1
        elif self.method == 'leitner' or self.method == 'pimsleur':
            pass
        elif self.method == 'lr':
            p, _ = self.predict(inst)
            err = p - inst.p
            for (k, x_k) in inst.fv:
                # rate = (1./(1+inst.p)) * self.lrate   / math.sqrt(1 + self.fcounts[k])
                rate = self.lrate / math.sqrt(1 + self.fcounts[k])
                # error update
                self.weights[k] -= rate * err * x_k
                # L2 regularization update
                self.weights[k] -= rate * self.l2wt * self.weights[k] / self.sigma**2
                # increment feature count for learning rate
                self.fcounts[k] += 1

    def train(self, trainset):
        if self.method == 'leitner' or self.method == 'pimsleur':
            return
        random.shuffle(trainset)
        for inst in trainset:
            self.train_update(inst)

    def losses(self, inst):
        p, h = self.predict(inst)
        slp = (inst.p - p)**2
        slh = (inst.h - h)**2
        return slp, slh, p, h

    def eval(self, testset, prefix=''):
        results = {'p': [], 'h': [], 'pp': [], 'hh': [], 'slp': [], 'slh': []}
        for inst in testset:
            slp, slh, p, h = self.losses(inst)
            results['p'].append(inst.p)     # ground truth
            results['h'].append(inst.h)
            results['pp'].append(p)         # predictions
            results['hh'].append(h)
            results['slp'].append(slp)      # loss function values
            results['slh'].append(slh)

        mae_all_ones = mae(results['p'], np.ones(len(results['p'])))
        print("mae_all_ones:", mae_all_ones, flush=True)

        mae_p = mae(results['p'], results['pp'])
        mae_h = mae(results['h'], results['hh'])
        cor_p = spearmanr(results['p'], results['pp'])
        cor_h = spearmanr(results['h'], results['hh'])
        total_slp = sum(results['slp'])
        total_slh = sum(results['slh'])
        total_l2 = sum([x**2 for x in self.weights.values()])
        total_loss = total_slp + self.hlwt*total_slh + self.l2wt*total_l2
        if prefix:
            sys.stderr.write('%s\t' % prefix)
        sys.stderr.write('%.1f (p=%.1f, h=%.1f, l2=%.1f)\tmae(p)=%.3f\tcor(p)=%.3f\tmae(h)=%.3f\tcor(h)=%.3f\n' % \
            (total_loss, total_slp, self.hlwt*total_slh, self.l2wt*total_l2, \
            mae_p, cor_p, mae_h, cor_h))

        return mae_p, cor_h

    def dump_weights(self, fname):
        with open(fname, 'w') as f:
            for (k, v) in self.weights.items():
                f.write('%s\t%.4f\n' % (k, v))

    def dump_predictions(self, fname, testset):
        with open(fname, 'w') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                f.write("\t".join([str(inst.p), str(pp), str(inst.h), str(hh), str(inst.lang), str(inst.uid), str(inst.ts)]) + "\n")

    def dump_detailed_predictions(self, fname, testset):
        with open(fname, 'w') as f:
            f.write('p\tpp\th\thh\tlang\tuser_id\ttimestamp\tlexeme_tag\n')
            for inst in testset:
                pp, hh = self.predict(inst)
                for i in range(inst.right):
                    f.write('1.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme))
                for i in range(inst.wrong):
                    f.write('0.0\t%.4f\t%.4f\t%.4f\t%s\t%s\t%d\t%s\n' % (pp, inst.h, hh, inst.lang, inst.uid, inst.ts, inst.lexeme))


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return min(max(p, 1e-6), 1 - 1e-6)


def hclip(h):
    # bound min/max half-life
    return min(max(h, MIN_HALF_LIFE), MAX_HALF_LIFE)


def mae(l1, l2):
    # mean average error
    return mean([abs(l1[i] - l2[i]) for i in range(len(l1))])


def mean(lst):
    # the average of a list
    return float(sum(lst))/len(lst)


def spearmanr(l1, l2):
    # spearman rank correlation
    m1 = mean(l1)
    m2 = mean(l2)
    num = 0.
    d1 = 0.
    d2 = 0.
    for i in range(len(l1)):
        num += (l1[i]-m1)*(l2[i]-m2)
        d1 += (l1[i]-m1)**2
        d2 += (l2[i]-m2)**2
    return num/math.sqrt(d1*d2)


def read_data(input_file, method, omit_bias=False, omit_lexemes=False, include_words=False, include_lemma=False,
              include_tags=False, include_subwords=False, nbest=1, include_complexity=False, include_lang=False,
              sim_key=None, sim_matrix=None, max_lines=None):
    # read learning trace data in specified format, see README for details
    sys.stderr.write('reading data...')
    sys.stderr.flush()
    instances = list()
    print(input_file, flush=True)
    if input_file.endswith('gz'):
        f = gzip.open(input_file, 'r')
    else:
        f = open(input_file, 'r', encoding="utf-8")
    reader = csv.DictReader(f)

    if include_lemma:
        historic_lemma = dict()

    # if include_subwords:
    #     historic_subwords = dict()
    if sim_matrix is not None:
        assert sim_matrix is not None
        user_history = dict()

    for i, row in enumerate(reader):
        if max_lines is not None and i >= max_lines:
            break
        p = pclip(float(row['p_recall']))
        t = float(row['delta'])/(60*60*24)  # convert time delta to days
        h = hclip(-t/(math.log(p, 2)))
        lang = '%s->%s' % (row['ui_language'], row['learning_language'])
        lexeme_id = row['lexeme_id']
        lexeme_string = row['lexeme_string']

        # break lexeme string into individual features
        lemma = lexeme_string.split("/")[1].split("<")[0]
        lexeme_word = lexeme_string.split("/")[0]
        if lexeme_word == "<*sf>":
            lexeme_word = lemma
        lexeme_tags = lexeme_string.split("/")[1].split("<")[1:]

        timestamp = int(row['timestamp'])
        user_id = row['user_id']

        seen = int(row['history_seen'])
        right = int(row['history_correct'])
        wrong = seen - right
        right_this = int(row['session_correct'])
        wrong_this = int(row['session_seen']) - right_this

        if sim_matrix is not None:
            if user_id not in user_history:
                user_history[user_id] = dict()
            user_history[user_id][lexeme_word] = (seen, right)

            sim_seen = 0
            sim_right = 0

            current_sim_index = sim_key[lexeme_word]
            for w in user_history[user_id]:
                sim_val = sim_matrix[current_sim_index][sim_key[w]]
                w_seen, w_right = user_history[user_id][w]
                sim_seen += w_seen * sim_val
                sim_right += w_right * sim_val

            sim_wrong = sim_seen - sim_right

        if include_lemma:
            user_lemma_key = (user_id, row['learning_language'], lemma)
            if user_lemma_key not in historic_lemma:
                historic_lemma[user_lemma_key] = {"seen": dict(), "right": dict()}
            historic_lemma[user_lemma_key]["seen"][lexeme_word] = seen
            historic_lemma[user_lemma_key]["right"][lexeme_word] = right

            seen_lemma = sum(historic_lemma[user_lemma_key]["seen"].values())
            right_lemma = sum(historic_lemma[user_lemma_key]["right"].values())
            wrong_lemma = seen_lemma - right_lemma

        if include_subwords:
            subword_skills = set()
            nbest_segmentations = lang_tokenizers[row['learning_language']].NBestEncodeAsPieces(lexeme_word, nbest)

            for segmentation in nbest_segmentations:
                subword_skills.update(segmentation)

            # for sw in subword_skills:
            #     if lexeme_word not in sw:
            #         user_subword_key = (user_id, row['learning_language'], sw)
            #         if user_subword_key not in historic_subwords:
            #             historic_subwords[user_subword_key] = {"seen": dict(), "right": dict()}
            #         historic_subwords[user_subword_key]["seen"][lexeme_word] = seen
            #         historic_subwords[user_subword_key]["right"][lexeme_word] = right

        if include_complexity:
            word_length = len(lexeme_word)
            lemma_length = len(lemma)
            word_lemma_distance = edit_distance(lemma, lexeme_word)

        # feature vector is a list of (feature, value) tuples
        fv = set()
        # core features based on method
        if method == 'leitner':
            fv.add((sys.intern('diff'), right-wrong))
        elif method == 'pimsleur':
            fv.add((sys.intern('total'), right+wrong))
        elif method == 'lr':
            fv.add((sys.intern('time'), t))
        else:
            # fv.add((sys.intern('right'), right))
            # fv.add((sys.intern('wrong'), wrong))
            if sim_matrix is not None:
                fv.add((sys.intern('sim_right'), math.sqrt(1+sim_right)))
                fv.add((sys.intern('sim_wrong'), math.sqrt(1+sim_wrong)))
            else:
                fv.add((sys.intern('right'), math.sqrt(1+right)))
                fv.add((sys.intern('wrong'), math.sqrt(1+wrong)))

            # optional flag features
            if not omit_bias:
                fv.add((sys.intern('bias'), 1.))
            if not omit_lexemes:
                fv.add((sys.intern('%s:%s' % (row['learning_language'], lexeme_string)), 1.))
            if include_words:
                fv.add((sys.intern('%s:%s' % (row['learning_language'], lexeme_word)), 1.))
            if include_lemma:
                fv.add((sys.intern('right_lemma'), math.sqrt(1+right_lemma)))
                fv.add((sys.intern('wrong_lemma'), math.sqrt(1+wrong_lemma)))
                fv.add((sys.intern('%s:%s' % (row['learning_language'], lemma)), 1.))
            if include_tags:
                for tag in lexeme_tags:
                    fv.add((sys.intern('%s:%s' % (row['learning_language'], tag)), 1.))
            if include_subwords:
                for sw in subword_skills:
                    fv.add((sys.intern('%s:%s' % (row['learning_language'], sw)), 1.))
            if include_complexity:
                fv.add((sys.intern('word_length'), word_length))
                fv.add((sys.intern('lemma_length'), lemma_length))
                fv.add((sys.intern('word_lemma_distance'), word_lemma_distance))
            if include_lang:
                fv.add((sys.intern('%s' % (lang)), 1.))

        fv = list(fv)

        instances.append(Instance(p, t, fv, h, (right+2.)/(seen+4.), lang, right_this, wrong_this, timestamp, user_id, lexeme_string))
        if i % 1000000 == 0:
            sys.stderr.write('%d...' % i)
            sys.stderr.flush()
    sys.stderr.write('done!\n')
    sys.stderr.flush()
    splitpoint = int(0.9 * len(instances))
    return instances[:splitpoint], instances[splitpoint:]


argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-b', action="store_true", default=False, help='omit bias feature')
argparser.add_argument('-l', action="store_true", default=False, help='omit lexeme features')
argparser.add_argument('-t', action="store_true", default=False, help='omit half-life term')

argparser.add_argument('--words', action="store_true", default=False, help='include words')
argparser.add_argument('--lemma', action="store_true", default=False, help='include lemma')
argparser.add_argument('--tags', action="store_true", default=False, help='include tags')
argparser.add_argument('--complexity', action="store_true", default=False, help='include word complexity measures')
argparser.add_argument('--src_tgt_lang', action="store_true", default=False, help='include source -> target languages')
argparser.add_argument('--subwords', action="store_true", default=False, help='include subword_skills')
argparser.add_argument('--tokenizer_dir', type=str, nargs='?', default='./', help='tokenizer directory')
argparser.add_argument('--vocab_size', type=int, nargs='?', default=5000, help='tokenizer vocab size')
argparser.add_argument('--nbest', type=int, nargs='?', default=2)
argparser.add_argument('--sim_file', type=str, nargs='?', default=None, help='similarity matrix file')

argparser.add_argument('--grid_search', action="store_true", default=False, help='grid search')
argparser.add_argument('-m', action="store", dest="method", default='hlr', help="hlr, lr, leitner, pimsleur")
argparser.add_argument('-x', action="store", dest="max_lines", type=int, default=None, help="maximum number of lines to read (for dev)")
argparser.add_argument('input_file', action="store", help='log file for training')


if __name__ == "__main__":

    args = argparser.parse_args()

    # model diagnostics
    sys.stderr.write('method = "%s"\n' % args.method)
    if args.b:
        sys.stderr.write('--> omit_bias\n')
    if args.l:
        sys.stderr.write('--> omit_lexemes\n')
    if args.t:
        sys.stderr.write('--> omit_h_term\n')

    if args.words:
        sys.stderr.write('--> include_words\n')
    if args.lemma:
        sys.stderr.write('--> include_lemma\n')
    if args.tags:
        sys.stderr.write('--> include_tags\n')
    if args.complexity:
        sys.stderr.write('--> include_complexity\n')
    if args.src_tgt_lang:
        sys.stderr.write('--> include_src_tgt_lang\n')

    if args.sim_file is not None:
        sys.stderr.write('--> use similarity matrix for counting right/wrong\n')

    sys.stderr.flush()

    lang_tokenizers = {}
    if args.subwords:
        sys.stderr.write('--> include_subwords (nbest='+str(args.nbest) + ', vocab_size=' + str(args.vocab_size) + ')\n')
        sys.stderr.flush()
        print("Loading sentence piece tokenizers...", flush=True)
        for path in glob.glob(args.tokenizer_dir + "**" + "_vocab=" + str(args.vocab_size) + "**.model"):
            sp_model = spm.SentencePieceProcessor()
            sp_model.Load(path)
            lang = path.split("/")[-1].split("_")[0]
            lang_tokenizers[lang] = sp_model

    if args.sim_file is not None:
        with open(args.sim_file + "_key.json", "r", encoding="utf-8") as key_file:
            word_to_id = json.load(key_file)

        with open(args.sim_file + ".npy", "rb") as matrix_file:
            sim_matrix = np.load(matrix_file)
    else:
        word_to_id = None
        sim_matrix = None

    # read data set
    trainset, testset = read_data(args.input_file, args.method, omit_bias=args.b, omit_lexemes=args.l,
                                  include_words=args.words, include_lemma=args.lemma, include_tags=args.tags,
                                  include_subwords=args.subwords, nbest=args.nbest, include_complexity=args.complexity,
                                  include_lang=args.src_tgt_lang, sim_key=word_to_id,
                                  sim_matrix=sim_matrix, max_lines=args.max_lines)
    sys.stderr.write('|train| = %d\n' % len(trainset))
    sys.stderr.write('|test|  = %d\n' % len(testset))
    sys.stderr.flush()

    # train model & print preliminary evaluation info
    if args.grid_search:
        grid_lrates = [0.001, 0.002, 0.003]
        grid_hlwts = [0.01, 0.02, 0.03]
        grid_l2wts = [0.1, 0.2, 0.3]

        for lrate in grid_lrates:
            for hlwt in grid_hlwts:
                for l2wt in grid_l2wts:
                    model = SpacedRepetitionModel(method=args.method, omit_h_term=args.t, lrate=lrate, hlwt=hlwt, l2wt=l2wt)

                    print("------ lrate:", lrate, "hlwt:", hlwt, "l2wt:", l2wt, "------", flush=True)
                    EPOCHS = 5
                    for epoch in range(EPOCHS):
                        model.train(trainset)
                        mae_p, cor_h = model.eval(testset, 'test')
                        print("epoch", epoch, ": mae_p =", mae_p, "cor_h =", cor_h, flush=True)

    else:
        lrate = .001
        hlwt = .01
        l2wt = .1

        model = SpacedRepetitionModel(method=args.method, omit_h_term=args.t, lrate=lrate, hlwt=hlwt, l2wt=l2wt)

        # EPOCHS = 5
        # for epoch in range(EPOCHS):
        model.train(trainset)
        model.eval(testset, 'test')

    # write out model weights and predictions
    filebits = [args.method] + \
        [k for k, v in sorted(vars(args).items()) if v is True] + \
        [os.path.splitext(os.path.basename(args.input_file).replace('.gz', ''))[0]]
    if args.max_lines is not None:
        filebits.append(str(args.max_lines))
    if args.sim_file is not None:
        filebits.append(str(args.sim_file))
    filebase = '.'.join(filebits)
    if not os.path.exists('results/'):
        os.makedirs('results/')
    model.dump_weights('results/'+filebase+'.weights')
    model.dump_predictions('results/'+filebase+'.preds', testset)
    # model.dump_detailed_predictions('results/'+filebase+'.detailed', testset)
