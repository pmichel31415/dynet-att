from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import dynet as dy
import time

import data
import options
import seq2seq

if __name__ == '__main__':
    # Retrieve options ==================================================
    opt = options.get_options()
    # Load data =========================================================
    if opt.verbose:
        print('Reading corpora')
    # Read vocabs
    if opt.dic_src:
        widss, ids2ws = data.load_dic(opt.dic_src)
    else:
        widss, ids2ws = data.read_dic(opt.train_src, max_size=opt.src_vocab_size)
        data.save_dic(opt.exp_name + '_src_dic.txt', widss)

    if opt.dic_dst:
        widst, ids2wt = data.load_dic(opt.dic_dst)
    else:
        widst, ids2wt = data.read_dic(opt.train_dst, max_size=opt.trg_vocab_size)
        data.save_dic(opt.exp_name + '_trg_dic.txt', widst)
    if opt.train:
        # Read training
        trainings_data = data.read_corpus(opt.train_src, widss)
        trainingt_data = data.read_corpus(opt.train_dst, widst)
        # Read validation
        valids_data = data.read_corpus(opt.valid_src, widss)
        validt_data = data.read_corpus(opt.valid_dst, widst)
    # Read test
    tests_data = data.read_corpus(opt.test_src, widss)
    testt_data = data.read_corpus(opt.test_dst, widst)

    # Create model ======================================================
    if opt.verbose:
        print('Creating model')
        sys.stdout.flush()
    s2s = seq2seq.Seq2SeqModel(opt.emb_dim,
                               opt.hidden_dim,
                               opt.att_dim,
                               widss,
                               widst,
                               model_file=opt.model,
                               bidir=opt.bidir,
                               word_emb=opt.word_emb,
                               dropout=opt.dropout_rate,
                               max_len=opt.max_len)

    if s2s.model_file is not None:
        s2s.load()
    s2s.model_file = opt.exp_name+'_model.txt'
    # Trainer ==========================================================
    if opt.train:
        if opt.trainer == 'sgd':
            trainer = dy.SimpleSGDTrainer(
                s2s.model, e0=opt.learning_rate, edecay=opt.learning_rate_decay)
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
    if opt.train:
        if opt.verbose:
            print('Creating batch loaders')
            sys.stdout.flush()
        trainbatchloader = data.BatchLoader(trainings_data, trainingt_data, opt.batch_size)
        devbatchloader = data.BatchLoader(valids_data, validt_data, opt.dev_batch_size)
    # Start training ====================================================
    if opt.train:
        if opt.verbose:
            print('starting training')
            sys.stdout.flush()
        start = time.time()
        train_loss = 0
        processed = 0
        best_dev_loss = np.inf
        i = 0
        for epoch in range(opt.num_epochs):
            for x, y in trainbatchloader:
                processed += sum(map(len, y))
                bsize = len(y)
                loss = s2s.calculate_loss(x, y)
                loss.backward()
                trainer.update()
                train_loss += loss.scalar_value() * bsize
                if (i+1) % opt.check_train_error_every == 0:
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
                    if dev_ppl < best_dev_loss:
                        best_dev_loss = dev_ppl
                        print('Best dev error up to date, saving model to', s2s.model_file)
                        s2s.save()
                    sys.stdout.flush()
                    start = time.time()

                if (i+1) % opt.test_every == 0:
                    print('Start running on test set, buckle up!')
                    sys.stdout.flush()
                    test_start = time.time()
                    f = open(opt.test_out, 'w+')
                    for x, y in zip(tests_data, testt_data):
                        y_hat = s2s.translate(x, decoding='beam_search', beam_size=opt.beam_size)
                        reference = [ids2wt[w] for w in y[1:-1]]
                        translation = [ids2wt[w] for w in y_hat[1:-1]]
                        print('t=', ' '.join(translation), file=f)
                    f.close()
                    test_elapsed = time.time()-test_start
                    print('Finished running on test set', test_elapsed, 'elapsed.')
                    sys.stdout.flush()
                    start = time.time()
                i = i+1
            trainer.update_epoch()
    # Start testing =====================================================
    if opt.test:
        print('Start running on test set, buckle up!')
        sys.stdout.flush()
        test_start = time.time()
        with open(opt.test_out, 'w+') as of:
            for x in tests_data:
                y = s2s.translate(x, decoding='beam_search', beam_size=opt.beam_size)
                translation = ' '.join([ids2wt[w] for w in y])
                of.write(translation+'\n')
        test_elapsed = time.time()-test_start
        print('Finished running on test set', test_elapsed, 'elapsed.')
        sys.stdout.flush()
