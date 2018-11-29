#!/usr/bin/env python3

import time

import numpy as np
import dynet as dy

import data
import seq2seq
import lm
import vocabulary
import utils



def load_pretrained_wembs(opt, vocab):
    if opt.pretrained_wembs is not None:
        print('Using pretrained word embeddings from %s' % opt.pretrained_wembs)
        wv = data.load_word_vectors(opt.pretrained_wembs, vocab.w2idt)
        d = wv.shape[1]
        pretrained_wembs = np.zeros((len(vocab.w2idt), opt.emb_dim))
        pretrained_wembs[:, (opt.emb_dim - d):] = wv
    else:
        pretrained_wembs = None
    return pretrained_wembs

def build_model(opt, lexicon, test=False):
    s2s = seq2seq.Seq2SeqModel(opt.num_layers,
                               opt.emb_dim,
                               opt.hidden_dim,
                               opt.att_dim,
                               lexicon,
                               model_file=opt.model,
                               enc_type=opt.encoder,
                               att_type=opt.attention,
                               dec_type=opt.decoder,
                               loss_type=opt.loss_type,
                               pretrained_wembs=load_pretrained_wembs(opt, lexicon),
                               label_smoothing=opt.label_smoothing,
                               kbest_synonyms=opt.kbest_synonyms,
                               dropout=opt.dropout_rate,
                               word_dropout=opt.word_dropout_rate,
                               unk_replacement=opt.unk_replacement,
                               log_unigram_bias=opt.log_unigram_bias,
                               mos_k=opt.mos_k,
                               desentencepiece=opt.desentencepiece,
                               max_len=opt.max_len)
    if test or opt.pretrained:
        if s2s.model_file is None:
            s2s.model_file = utils.exp_filename(opt, 'model')
        print('loading pretrained model from %s' % s2s.model_file)
        s2s.load()
    else:
        if s2s.model_file is not None:
            s2s.load()
        s2s.model_file = utils.exp_filename(opt, 'model')
    return s2s


def get_vocab(opt):
    load = not opt.train
    if opt.vocab_file is None:
        opt.vocab_file = utils.exp_filename(opt, 'vocab_file')
    else:
        load = True
    if opt.train and not load:
        vocab = vocabulary.Vocabulary()
        vocab.init(opt)
        vocabulary.save_vocab(opt.vocab_file, vocab)
    else:
        if opt.vocab_file is None:
            opt.vocab_file = utils.exp_filename(opt, 'vocab_file')
        print('Loading vocabulary from file: %s' % opt.vocab_file)
        vocab = vocabulary.load_vocab(opt.vocab_file)
    return vocab


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
            opt.lm_file = utils.exp_filename(opt, 'lm')
        lang_model.load(opt.lm_file)
    else:
        print('training lm')
        lang_model.fit(train_data)
        opt.lm_file = utils.exp_filename(opt, 'lm')
        lang_model.save(opt.lm_file)
    return lang_model


def get_trainer(opt, s2s):
    if opt.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(s2s.pc,
                                      learning_rate=opt.learning_rate)
    elif opt.trainer == 'clr':
        trainer = dy.CyclicalSGDTrainer(s2s.pc,
                                        learning_rate_min=opt.learning_rate / 10.0,
                                        learning_rate_max=opt.learning_rate)
    elif opt.trainer == 'momentum':
        trainer = dy.MomentumSGDTrainer(s2s.pc,
                                        learning_rate=opt.learning_rate)
    elif opt.trainer == 'rmsprop':
        trainer = dy.RMSPropTrainer(s2s.pc,
                                    learning_rate=opt.learning_rate)
    elif opt.trainer == 'adam':
        trainer = dy.AdamTrainer(s2s.pc,
                                 opt.learning_rate)
    else:
        print('Trainer name invalid or not provided, using SGD', file=sys.stderr)
        trainer = dy.SimpleSGDTrainer(s2s.pc,
                                      learning_rate=opt.learning_rate)

    trainer.set_clip_threshold(opt.gradient_clip)

    return trainer
