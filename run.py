from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import options

import numpy as np
import dynet as dy

import data
import seq2seq
import evaluation
import helpers

def train(opt):
    log = helpers.Logger(opt.verbose)
    timer = helpers.Timer()
    # Load data =========================================================
    log.info('Reading corpora')
    # Read vocabs
    widss, ids2ws, widst, ids2wt = helpers.get_dictionaries(opt)

    # Read training
    trainings_data = data.read_corpus(opt.train_src, widss)
    trainingt_data = data.read_corpus(opt.train_dst, widst)
    # Read validation
    valids_data = data.read_corpus(opt.valid_src, widss)
    validt_data = data.read_corpus(opt.valid_dst, widst)
    # Validation output
    if not opt.valid_out:
        opt.valid_out = helpers.exp_filename(opt, 'valid.out')
    # Get target language model
    lang_model = helpers.get_language_model(opt, trainingt_data, widst) 
    # Create model ======================================================
    log.info('Creating model')
    s2s = helpers.build_model(opt, widss, widst, lang_model)

    # Trainer ==========================================================
    trainer = helpers.get_trainer(opt, s2s)
    log.info('Using '+opt.trainer+' optimizer')
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(widss), trg_dict_size=len(widst))
    # Creat batch loaders ===============================================
    log.info('Creating batch loaders')
    trainbatchloader = data.BatchLoader(trainings_data, trainingt_data, opt.batch_size)
    devbatchloader = data.BatchLoader(valids_data, validt_data, opt.dev_batch_size)
    # Start training ====================================================
    log.info('starting training')
    timer.restart()
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
                trainer.status()
                log.info(" Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                      (logloss, ppl, timer.tick(), processed))
                train_loss = 0
                processed = 0
            if (i+1) % opt.check_valid_error_every == 0:
                # Check generalization error on the validation set from time to time
                dev_loss = 0
                dev_processed = 0
                timer.restart()
                for x, y in devbatchloader:
                    dev_processed += sum(map(len, y))
                    bsize = len(y)
                    loss = s2s.calculate_loss(x, y, test=True)
                    dev_loss += loss.scalar_value() * bsize
                dev_logloss = dev_loss/dev_processed
                dev_ppl = np.exp(dev_logloss)
                log.info("[epoch %d] Dev loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                      (epoch, dev_logloss, dev_ppl, timer.tick(), dev_processed))

            if (i+1) % opt.valid_bleu_every == 0:
                # Check BLEU score on the validation set from time to time
                log.info('Start translating validation set, buckle up!')
                timer.restart()
                with open(opt.valid_out, 'w+') as f:
                    for x in valids_data:
                        y_hat = s2s.translate(x, beam_size=opt.beam_size)
                        translation = [ids2wt[w] for w in y_hat[1:-1]]
                        print(' '.join(translation), file=f)
                bleu, details = evaluation.bleu_score(opt.valid_dst, opt.valid_out)
                log.info('Finished translating validation set %.2f elapsed.' % timer.tick())
                log.info(details)
                # Early stopping : save the latest best model
                if bleu > best_bleu:
                    best_bleu = bleu
                    log.info('Best BLEU score up to date, saving model to %s' % s2s.model_file)
                    s2s.save()
                    deadline = 0
                else:
                    deadline += 1
                if opt.patience > 0 and deadline > opt.patience:
                    log.info('No improvement since %d epochs, early stopping with best validation BLEU score: %.3f' % (deadline, best_bleu))
                    exit()
            i = i + 1
        trainer.update_epoch()


def test(opt):
    log = helpers.Logger(opt.verbose)
    timer = helpers.Timer()
    # Load data =========================================================
    log.info('Reading corpora')
    # Read vocabs
    widss, ids2ws, widst, ids2wt = helpers.get_dictionaries(opt, test=True)
    # Read test
    tests_data = np.asarray(data.read_corpus(opt.test_src, widss), dtype=list)
    # Test output
    if not opt.test_out:
        opt.test_out = helpers.exp_filename(opt,'test.out')
    # Get target language model
    lang_model = helpers.get_language_model(opt, trainingt_data, widst, test=True) 
    # Create model ======================================================
    log.info('Creating model')
    s2s = helpers.build_model(opt, widss, widst, lang_model)
    # Print configuration ===============================================
    if opt.verbose:
        options.print_config(opt, src_dict_size=len(widss), trg_dict_size=len(widst))
    # Start testing =====================================================
    log.info('Start running on test set, buckle up!')
    timer.restart()
    translations = []
    for i, x in enumerate(tests_data):
	y = s2s.translate(x, beam_size=opt.beam_size)
	translations.append(' '.join([ids2wt[w] for w in y[1:-1]]))
    np.savetxt(opt.test_out, translations, fmt='%s')
    translations = np.asarray(translations, dtype=str)
    BLEU, details = evaluation.bleu_score(opt.test_dst, opt.test_out)
    log.info('Finished running on test set %.2f elapsed.' % timer.tick())
    log.info(details)


if __name__ == '__main__':
    # Retrieve options ==================================================
    opt = options.get_options()
    if opt.train:
        train(opt)
    elif opt.test:
        test(opt)
