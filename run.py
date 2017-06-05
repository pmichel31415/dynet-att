from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import options

import numpy as np
import dynet as dy
import time

import data
import seq2seq
import evaluation


def train(opt):
    # Load data =========================================================
    if opt.verbose:
        print('Reading corpora')
    # Read vocabs
    if opt.dic_src:
        widss, ids2ws = data.load_dic(opt.dic_src)
    else:
        widss, ids2ws = data.read_dic(opt.train_src, max_size=opt.src_vocab_size, min_freq=opt.min_freq)
        data.save_dic(opt.exp_name + '_src_dic.txt', widss)

    if opt.dic_dst:
        widst, ids2wt = data.load_dic(opt.dic_dst)
    else:
        widst, ids2wt = data.read_dic(opt.train_dst, max_size=opt.trg_vocab_size, min_freq=opt.min_freq)
        data.save_dic(opt.exp_name + '_trg_dic.txt', widst)

    # Read training
    trainings_data = data.read_corpus(opt.train_src, widss)
    trainingt_data = data.read_corpus(opt.train_dst, widst)
    # Read validation
    valids_data = data.read_corpus(opt.valid_src, widss)
    validt_data = data.read_corpus(opt.valid_dst, widst)
    # Validation output
    if not opt.valid_out:
        opt.valid_out = opt.output_dir + '/' + opt.exp_name + '.valid.out'
    
    # Create model ======================================================
    if opt.verbose:
        print('Creating model')
        sys.stdout.flush()
    s2s = seq2seq.Seq2SeqModel(opt.num_layers,
                               opt.emb_dim,
                               opt.hidden_dim,
                               opt.att_dim,
                               widss,
                               widst,
                               model_file=opt.model,
                               bidir=opt.bidir,
                               word_emb=opt.word_emb,
                               dropout=opt.dropout_rate,
                               word_dropout=opt.word_dropout_rate,
                               max_len=opt.max_len)

    if s2s.model_file is not None:
        s2s.load()
    s2s.model_file = opt.exp_name+'_model.txt'
    # Trainer ==========================================================
    if opt.trainer == 'sgd':
        trainer = dy.SimpleSGDTrainer(
            s2s.model, e0=opt.learning_rate, edecay=opt.learning_rate_decay)
    if opt.trainer == 'clr':
        trainer = dy.CyclicalSGDTrainer(s2s.model, e0_min=opt.learning_rate / 10.0,
                                        e0_max=opt.learning_rate, edecay=opt.learning_rate_decay)
    elif opt.trainer == 'momentum':
        trainer = dy.MomentumSGDTrainer(
            s2s.model, e0=opt.learning_rate, edecay=opt.learning_rate_decay)
    elif opt.trainer == 'rmsprop':
        trainer = dy.RMSPropTrainer(s2s.model, e0=opt.learning_rate,
                                    edecay=opt.learning_rate_decay)
    elif opt.trainer == 'adam':
        trainer = dy.AdamTrainer(s2s.model, opt.learning_rate, edecay=opt.learning_rate_decay)
    else:
        print('Trainer name invalid or not provided, using SGD', file=sys.stderr)
        trainer = dy.SimpleSGDTrainer(
            s2s.model, e0=opt.learning_rate, edecay=opt.learning_rate_decay)
    if opt.verbose:
        print('Using '+opt.trainer+' optimizer')
    trainer.set_clip_threshold(opt.gradient_clip)
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(widss), trg_dict_size=len(widst))
        sys.stdout.flush()
    # Creat batch loaders ===============================================
    if opt.verbose:
        print('Creating batch loaders')
        sys.stdout.flush()
    trainbatchloader = data.BatchLoader(trainings_data, trainingt_data, opt.batch_size)
    devbatchloader = data.BatchLoader(valids_data, validt_data, opt.dev_batch_size)
    # Start training ====================================================
    if opt.verbose:
        print('starting training')
        sys.stdout.flush()
    start = time.time()
    train_loss = 0
    processed = 0
    best_bleu = -1
    deadline = 0
    i = 0
    for epoch in range(opt.num_epochs):
        for x, y in trainbatchloader:
            processed += sum(map(len, y))
            bsize = len(y)
            # Compute loss
            loss = s2s.calculate_loss(x, y)
            # Backward pass and parameter update
            loss.backward()
            trainer.update()
            train_loss += loss.scalar_value() * bsize
            if (i+1) % opt.check_train_error_every == 0:
                # Check average training error from time to time
                logloss = train_loss / processed
                ppl = np.exp(logloss)
                elapsed = time.time()-start
                trainer.status()
                print(" Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                      (logloss, ppl, elapsed, processed))
                start = time.time()
                train_loss = 0
                processed = 0
                sys.stdout.flush()
            if (i+1) % opt.check_valid_error_every == 0:
                # Check generalization error on the validation set from time to time
                dev_loss = 0
                dev_processed = 0
                dev_start = time.time()
                for x, y in devbatchloader:
                    dev_processed += sum(map(len, y))
                    bsize = len(y)
                    loss = s2s.calculate_loss(x, y, test=True)
                    dev_loss += loss.scalar_value() * bsize
                dev_logloss = dev_loss/dev_processed
                dev_ppl = np.exp(dev_logloss)
                dev_elapsed = time.time()-dev_start
                print("[epoch %d] Dev loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                      (epoch, dev_logloss, dev_ppl, dev_elapsed, dev_processed))
                sys.stdout.flush()
                start = time.time()

            if (i+1) % opt.valid_bleu_every == 0:
                # Check BLEU score on the validation set from time to time
                print('Start translating validation set, buckle up!')
                sys.stdout.flush()
                bleu_start = time.time()
                with open(opt.valid_out, 'w+') as f:
                    for x in valids_data:
                        y_hat = s2s.translate(x, beam_size=opt.beam_size)
                        translation = [ids2wt[w] for w in y_hat[1:-1]]
                        print(' '.join(translation), file=f)
                bleu, details = evaluation.bleu_score(opt.valid_dst, opt.valid_out)
                bleu_elapsed = time.time()-bleu_start
                print('Finished translating validation set', bleu_elapsed, 'elapsed.')
                print(details)
                # Early stopping : save the latest best model
                if bleu > best_bleu:
                    best_bleu = bleu
                    print('Best BLEU score up to date, saving model to', s2s.model_file)
                    s2s.save()
                    deadline = 0
                else:
                    deadline += 1
                if opt.patience > 0 and deadline > opt.patience:
                    print('No improvement since',deadline,'epochs, early stopping with best validation BLEU score:', best_bleu)
                    exit()
                sys.stdout.flush()
                start = time.time()
            i = i+1
        trainer.update_epoch()


def test(opt):
    # Load data =========================================================
    if opt.verbose:
        print('Reading corpora')
    # Read vocabs
    if opt.dic_src:
        widss, ids2ws = data.load_dic(opt.dic_src)
    elif opt.train_src:
        widss, ids2ws = data.read_dic(opt.train_src, max_size=opt.src_vocab_size, min_freq=opt.min_freq)
        data.save_dic(opt.exp_name + '_src_dic.txt', widss)
    else:
        widss, ids2ws = data.load_dic(opt.exp_name + '_src_dic.txt')

    if opt.dic_dst:
        widst, ids2wt = data.load_dic(opt.dic_dst)
    elif opt.train_dst:
        widst, ids2wt = data.read_dic(opt.train_dst, max_size=opt.trg_vocab_size, min_freq=opt.min_freq)
        data.save_dic(opt.exp_name + '_trg_dic.txt', widst)
    else:
        widst, ids2wt = data.load_dic(opt.exp_name + '_trg_dic.txt')
    # Read test
    tests_data = np.asarray(data.read_corpus(opt.test_src, widss), dtype=list)
    # Test output
    if not opt.test_out:
        opt.test_out = opt.output_dir + '/' + opt.exp_name + '.test.out'
    # Create model ======================================================
    if opt.verbose:
        print('Creating model')
        sys.stdout.flush()
    s2s = seq2seq.Seq2SeqModel(opt.num_layers,
                               opt.emb_dim,
                               opt.hidden_dim,
                               opt.att_dim,
                               widss,
                               widst,
                               model_file=opt.model,
                               bidir=opt.bidir,
                               word_emb=opt.word_emb,
                               dropout=opt.dropout_rate,
                               max_len=opt.max_len)

    if s2s.model_file is None:
        s2s.model_file = opt.exp_name + '_model.txt'
    s2s.load()
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(widss), trg_dict_size=len(widst))
        sys.stdout.flush()
    # Start testing =====================================================
    print('Start running on test set, buckle up!')
    sys.stdout.flush()
    test_start = time.time()
    translations = []
    for i, x in enumerate(tests_data):
        y = s2s.translate(x, beam_size=opt.beam_size)
        translations.append(' '.join([ids2wt[w] for w in y[1:-1]]))
    np.savetxt(opt.test_out, translations, fmt='%s')
    BLEU, details = evaluation.bleu_score(opt.test_dst, opt.test_out)
    test_elapsed = time.time()-test_start
    print('Finished running on test set', test_elapsed, 'elapsed.')
    print(details)
    sys.stdout.flush()
    bleus = []
    gold_file = opt.test_out[:-4] + '_gold.txt'
    hyp_file = opt.test_out[:-4] + '_boot.txt'
    translations = np.asarray(translations, dtype=str)
    gold = np.loadtxt(opt.test_dst, dtype=str, delimiter='\n')
    for k in range(opt.bootstrap_number):
        if opt.bootstrap_size < 100:
            subset = np.random.choice(len(tests_data), int(opt.bootstrap_size * len(tests_data) / 100.0)).astype(int)
        else:
            subset = np.arange(len(tests_data), dtype=int)
        np.savetxt(hyp_file, translations[subset], fmt='%s')
        np.savetxt(gold_file, gold[subset], fmt='%s')
        BLEU, details = evaluation.bleu_score(gold_file, hyp_file)
        bleus.append(BLEU)
        print(details)
        sys.stdout.flush()
    print('Confidence interval 5%% - 95%% : %3.f - %3.f' % (np.percentile(bleus, 5),np.percentile(bleus, 5)))
    np.savetxt(opt.output_dir + '/bleus_' + opt.exp_name + '.txt', bleus)



if __name__ == '__main__':
    # Retrieve options ==================================================
    opt = options.get_options()
    if opt.train:
        train(opt)
    elif opt.test:
        test(opt)
