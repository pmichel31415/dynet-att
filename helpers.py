from __future__ import print_function, division

import time

import dynet as dy

import data
import seq2seq
import lm

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


class Logger(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, string):
        if self.verbose:
            print(string)
        sys.stdout.flush()


class Timer(object):

    def __init__(self, verbose=False):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def tick(self):
        elapsed = self.elapsed()
        self.restart()
        return elapsed


def exp_filename(opt, name):
    return opt.output_dir + '/' + opt.exp_name + '_' + name


def build_model(opt, widss, widst, lang_model, test=False):
    s2s = seq2seq.Seq2SeqModel(opt.num_layers,
                               opt.emb_dim,
                               opt.hidden_dim,
                               opt.att_dim,
                               widss,
                               widst,
                               model_file=opt.model,
                               enc_type=opt.encoder,
                               att_type=opt.attention,
                               dec_type=opt.decoder,
                               lang_model=lang_model,
                               label_smoothing=opt.label_smoothing,
                               dropout=opt.dropout_rate,
                               word_dropout=opt.word_dropout_rate,
                               max_len=opt.max_len)
    if test:
        if s2s.model_file is None:
            s2s.model_file = exp_filename(opt, 'model')
        s2s.load()
    else:
        if s2s.model_file is not None:
            s2s.load()
        s2s.model_file = exp_filename(opt, 'model')
    return s2s


def get_language_model(opt, train_data, w2id, test=False):
    if opt.language_model is None:
        return None
    if opt.language_model == 'uniform':
        return None
    elif opt.language_model == 'unigram':
        lang_model = lm.UnigramLanguageModel(w2id)
    elif opt.language_model == 'bigram':
        lang_model = lm.BigramLanguageModel(w2id)
    else:
        print('Unknown language model %s, using unigram language model' % opt.language_model)
        lang_model = lm.UnigramLanguageModel(w2id)

    if opt.lm_file is not None or test:
        if opt.lm_file is None:
            opt.lm_file = exp_filename(opt, 'lm')
        lang_model.load(opt.lm_file)
    else:
        print('training lm')
        lang_model.fit(train_data)
        opt.lm_file = exp_filename(opt, 'lm')
        lang_model.save(opt.lm_file)
    return lang_model


def get_dictionaries(opt, test=False):
    if opt.dic_src:
        widss, ids2ws = data.load_dic(opt.dic_src)
    elif opt.train_src or not test:
        widss, ids2ws = data.read_dic(
            opt.train_src, max_size=opt.src_vocab_size, min_freq=opt.min_freq)
        data.save_dic(exp_filename(opt, 'src_dic'), widss)
    else:
        widss, ids2ws = data.load_dic(exp_filename(opt, 'src_dic'))

    if opt.dic_dst:
        widst, ids2wt = data.load_dic(opt.dic_dst)
    elif opt.train_dst or not test:
        widst, ids2wt = data.read_dic(
            opt.train_dst, max_size=opt.trg_vocab_size, min_freq=opt.min_freq)
        data.save_dic(exp_filename(opt, 'trg_dic'), widst)
    else:
        widst, ids2wt = data.load_dic(exp_filename(opt, 'trg_dic'))

    return widss, ids2ws, widst, ids2wt


def get_trainer(opt, s2s):
    if opt.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(s2s.pc,
                                      e0=opt.learning_rate,
                                      edecay=opt.learning_rate_decay)
    elif opt.trainer == 'clr':
        trainer = dy.CyclicalSGDTrainer(s2s.pc,
                                        e0_min=opt.learning_rate / 10.0,
                                        e0_max=opt.learning_rate,
                                        edecay=opt.learning_rate_decay)
    elif opt.trainer == 'momentum':
        trainer = dy.MomentumSGDTrainer(s2s.pc,
                                        e0=opt.learning_rate,
                                        edecay=opt.learning_rate_decay)
    elif opt.trainer == 'rmsprop':
        trainer = dy.RMSPropTrainer(s2s.pc,
                                    e0=opt.learning_rate,
                                    edecay=opt.learning_rate_decay)
    elif opt.trainer == 'adam':
        trainer = dy.AdamTrainer(s2s.pc,
                                 opt.learning_rate,
                                 edecay=opt.learning_rate_decay)
    else:
        print('Trainer name invalid or not provided, using SGD', file=sys.stderr)
        trainer = dy.SimpleSGDTrainer(s2s.pc,
                                      e0=opt.learning_rate,
                                      edecay=opt.learning_rate_decay)

    trainer.set_clip_threshold(opt.gradient_clip)

    return trainer
