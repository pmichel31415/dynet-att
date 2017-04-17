# rnn LM

from __future__ import print_function, division
import numpy as np
import argparse
from collections import defaultdict
import dynet as dy
import time
import pickle
import sys

CHECK_TRAIN_ERROR_EVERY = 10
CHECK_VALID_ERROR_EVERY = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--dynet-seed", default=0, type=int)
parser.add_argument("--dynet-mem", default=512, type=int)
parser.add_argument("--dynet-gpus", default=0, type=int)
parser.add_argument("--train_src", '-ts',
                    default='en-de/train.en-de.en', type=str)
parser.add_argument("--train_dst", '-td',
                    default='en-de/train.en-de.de', type=str)
parser.add_argument("--valid_src", '-vs',
                    default='en-de/valid.en-de.en', type=str)
parser.add_argument("--valid_dst", '-vd',
                    default='en-de/valid.en-de.de', type=str)
parser.add_argument("--test_src", '-tes',
                    default='en-de/test.en-de.en', type=str)
parser.add_argument("--test_out", '-teo',
                    default='results/test.en-de.de', type=str)
parser.add_argument("--model", '-m', type=str, help='Model to load from')
parser.add_argument('--num_epochs', '-ne',
                    type=int, help='Number of epochs', default=1)
parser.add_argument('--src_vocab_size', '-svs',
                    type=int, help='Maximum vocab size of the source language', default=40000)
parser.add_argument('--trg_vocab_size', '-tvs',
                    type=int, help='Maximum vocab size of the target language', default=20000)
parser.add_argument('--batch_size', '-bs',
                    type=int, help='minibatch size', default=20)
parser.add_argument('--dev_batch_size', '-dbs',
                    type=int, help='minibatch size for the validation set', default=10)
parser.add_argument('--emb_dim', '-de',
                    type=int, help='embedding size', default=256)
parser.add_argument('--hidden_dim', '-dh',
                    type=int, help='hidden size', default=256)
parser.add_argument('--dropout_rate', '-dr',
                    type=float, help='dropout rate', default=0.0)
parser.add_argument('--learning_rate', '-lr',
                    type=float, help='learning rate', default=1.0)
parser.add_argument('--learning_rate_decay', '-lrd',
                    type=float, help='learning rate decay', default=0.0)
parser.add_argument('--check_train_error_every', '-ct',
                    type=int, help='Check train error every', default=100)
parser.add_argument('--check_valid_error_every', '-cv',
                    type=int, help='Check valid error every', default=1000)
parser.add_argument('--test_every', '-te',
                    type=int, help='Run on test set every', default=500)
parser.add_argument('--max_len', '-ml', type=int,
                    help='Maximum length of generated sentences', default=60)
parser.add_argument('--beam_size', '-bm', type=int,
                    help='Beam size for beam search', default=1)
parser.add_argument("--exp_name", '-en', type=str, required=True,
                    help='Name of the experiment (used so save the model)')
parser.add_argument("--bidir", '-bid',
                    help="Activates bidirectionnal encoding",
                    action="store_true")
parser.add_argument("--word_emb", '-we',
                    help="Activates direct word embedding for attention",
                    action="store_true")
parser.add_argument("--verbose", '-v',
                    help="increase output verbosity",
                    action="store_true")
parser.add_argument("--debug", '-dbg',
                    help="Print debugging info",
                    action="store_true")
parser.add_argument("--train",
                    help="Print debugging info",
                    action="store_true")
parser.add_argument("--test",
                    help="Print debugging info",
                    action="store_true")


args = parser.parse_args()

verbose = args.verbose
debug = args.debug

# Writing a function to read in the training and test corpora, and
# converting the words into numerical IDs.


def reverse_dic(dic):
    rev_dic = dict()
    for k, v in dic.items():
        rev_dic[v] = k
    return rev_dic

def read_dic(file,max_size=20000,min_freq=1):
    dic = defaultdict(lambda: 0)
    freqs = defaultdict(lambda: 0)
    dic['UNK'],dic['SOS'],dic['EOS']=0,1,2
    with open(file, 'r') as f:
        for l in f:
            sent = l.strip().split()
            for word in sent:
                freqs[word] += 1

    sorted_words = sorted(freqs.iteritems(), key=lambda x : x[1], reverse=True)
    for i in range(max_size):
        word, freq = sorted_words[i]
        if freq<=min_freq:
            continue
        dic[word]=len(dic)

    return dic, reverse_dic(dic)

def read_corpus(file, dic):
    # for each line in the file, split the words and turn them into IDs like
    # this:
    sentences = []
    with open(file, 'r') as f:
        for l in f:
            sent = [dic['SOS']]
            for w in l.split():
                if w not in dic:
                    sent.append(dic['UNK'])
                else:
                    sent.append(dic[w])
            sent.append(dic['EOS'])
            sentences.append(sent)
    return sentences


def print_config():
    print('======= CONFIG =======')
    for k, v in vars(args).items():
        print(k, ':', v)
    print('Source vocabulary size :', len(widss))
    print('Target vocabulary size :', len(widst))
    print('======================')


class BatchLoader(object):

    def __init__(self, datas, datat, bsize):
        self.batches=[]

        self.bs = bsize

        # Bucket samples by source sentence length
        buckets=defaultdict(list)
        for src,trg in zip(datas,datat):
            buckets[len(src)].append((src,trg))
        
        for src_len, bucket in buckets.iteritems():
            np.random.shuffle(bucket)
            num_batches = int(np.ceil(len(bucket) * 1.0 / self.bs))
            for i in range(num_batches):
                cur_batch_size = self.bs if i < num_batches - 1 else len(bucket) - self.bs * i
                self.batches.append(([bucket[i * self.bs + j][0] for j in range(cur_batch_size)],
                               [bucket[i * self.bs + j][1] for j in range(cur_batch_size)]))

        self.n = len(self.batches)
        self.reseed()

    def reseed(self):
        print('Reseeding the dataset')
        self.i = 0
        np.random.shuffle(self.batches)

    def next(self):
        if self.i >= self.n - 1:
            self.reseed()
            raise StopIteration()
        self.i += 1
        return self.batches[self.i]

    def __next__(self):
        return self.next()

    def __iter__(self): return self


class Seq2SeqModel(dy.Saveable):

    def __init__(self,
                 model,
                 input_dim,
                 hidden_dim,
                 source_vocab_size,
                 target_vocab_size,
                 bidir=False,
                 word_emb=False,
                 dropout=0.0,
                 max_len=60):
        # Store config
        self.di, self.dh = input_dim, hidden_dim
        self.vs, self.vt = source_vocab_size, target_vocab_size
        self.bidir = bidir
        self.word_emb = word_emb
        self.dr = dropout

        # Declare parameters
        self.enc = dy.GRUBuilder(1, self.di, self.dh, model)
        self.rev_enc = dy.GRUBuilder(1, self.di, self.dh, model)
        self.dec_di = self.di + self.dh + \
            (self.dh if self.bidir else 0) + (self.di if self.word_emb else 0)
        self.dec = dy.GRUBuilder(1, self.dec_di, self.dh, model)
        self.A_p = model.add_parameters((self.dh, self.dec_di - self.di))
        self.MS_p = model.add_lookup_parameters((self.vs, self.di))
        self.MT_p = model.add_lookup_parameters((self.vt, self.di))
        self.D_di = 2 * self.dh + (self.dh if self.bidir else 0) + \
            (self.di if self.word_emb else 0)
        self.D_p = model.add_parameters((self.vt, self.D_di))
        self.b_p = model.add_parameters((self.vt,))

        self.max_len = max_len

    def calculate_loss(self, src, trg):
        dy.renew_cg()

        bsize = len(src)

        input_len = max(len(s) for s in src)
        output_len = max(len(s) for s in trg)
        # Pad
        if debug:
            start = time.time()
        x = np.zeros((input_len, bsize), dtype=int)
        masksx = np.ones((input_len, self.di, bsize), dtype=float)
        for i in range(bsize):
            srci = src[i][:]
            masksx[len(srci):, i] = 0.0
            while len(srci) < input_len:
                srci.append(widss['EOS'])
            x[:, i] = srci
        y = np.zeros((output_len, bsize), dtype=int)

        masksy = np.ones((output_len, bsize), dtype=float)
        for i in range(bsize):
            trgi = trg[i][:]
            masksy[len(trgi):, i] = 0.0
            while len(trgi) < output_len:
                trgi.append(widst['EOS'])
            y[:, i] = trgi

        D = dy.parameter(self.D_p)
        b = dy.parameter(self.b_p)
        A = dy.parameter(self.A_p)

        # Set dropout if necessary
        self.enc.set_dropout(self.dr)
        self.dec.set_dropout(self.dr)
        es = self.enc.initial_state()
        ds = self.dec.initial_state()
        errs = []
        encoded_states = []
        # Encode
        if debug:
            elapsed = time.time()-start
            print('Preprocessing took : ', elapsed)
            start = time.time()

        wembs = [dy.lookup_batch(self.MS_p, iw) for iw in x]

        encoded_states = es.transduce(wembs)

        if self.bidir:
            self.rev_enc.set_dropout(self.dr)
            res = self.rev_enc.initial_state()
            rev_encoded_states = res.transduce(wembs[::-1])[::-1]

        if debug:
            elapsed = time.time()-start
            print('Building encoding : ', elapsed)
            start = time.time()
        # Attend
        H = dy.concatenate_cols(encoded_states)
        if self.bidir:
            H_bidir = dy.concatenate_cols(rev_encoded_states)
            H = dy.concatenate([H, H_bidir])
        if self.word_emb:
            H_word_embs = dy.concatenate_cols(wembs)
            H = dy.concatenate([H, H_word_embs])
        if debug:
            elapsed = time.time()-start
            print('Building attention : ', elapsed)
            start = time.time()
        # Decode
        # Initialize decoder
        start = dy.lookup_batch(self.MT_p, [widst['SOS']] * bsize)
        start = dy.concatenate([start, dy.zeroes((self.dec_di-self.di,), batch_size=bsize)])
        ds = ds.add_input(start)
        # Compute context
        h = ds.output()
        context = H * dy.softmax(dy.transpose(A * H) * h)
        # Loop
        for cw, nw, mask in zip(y, y[1:], masksy[1:]):
            embs = dy.lookup_batch(self.MT_p, cw)
            # Run LSTM
            ds = ds.add_input(dy.concatenate([embs, context]))
            # Compute next context
            h = ds.output()
            context = H * dy.softmax(dy.transpose(A * H) * h)
            # Get distribution over words
            s = dy.affine_transform([b,D,dy.concatenate([h, context])])
            masksy_e = dy.inputTensor(mask, batched=True)
            err = dy.cmult(dy.pickneglogsoftmax_batch(s, nw), masksy_e)
            errs.append(err)
        if debug:
            elapsed = time.time()-start
            print('Building decoding : ', elapsed)
        err = dy.sum_batches(dy.esum(errs))

        return err

    def translate(self, x, decoding='greedy', T=1.0, beam_size=1):
        dy.renew_cg()
        input_len = len(x)
        D = dy.parameter(self.D_p)
        b = dy.parameter(self.b_p)
        A = dy.parameter(self.A_p)
        self.enc.disable_dropout()
        self.dec.disable_dropout()
        es = self.enc.initial_state()
        ds = self.dec.initial_state()
        encoded_states = []
        # Encode
        if debug:
            start = time.time()
        
        wembs = [dy.lookup(self.MS_p, iw) for iw in x]

        encoded_states = es.transduce(wembs)

        if self.bidir:
            self.rev_enc.set_dropout(self.dr)
            res = self.rev_enc.initial_state()
            rev_encoded_states = res.transduce(wembs[::-1])[::-1]

        if debug:
            elapsed = time.time()-start
            print('Building encoding : ', elapsed)
            start = time.time()
        # Attend
        H = dy.concatenate_cols(encoded_states)
        if self.bidir:
            H_bidir = dy.concatenate_cols(rev_encoded_states)
            H = dy.concatenate([H, H_bidir])
        if self.word_emb:
            H_word_embs = dy.concatenate_cols(encoded_wembs)
            H = dy.concatenate([H, H_word_embs])
        if debug:
            elapsed = time.time()-start
            print('Building attention : ', elapsed)
            start = time.time()
        # Decode
        cw = widst['SOS']
        words = []
        beam = []
        start = dy.lookup(self.MT_p, widst['SOS'])
        start = dy.concatenate([start, dy.zeroes((self.dec_di - self.di,))])
        ds = ds.add_input(start)
        # Compute context
        h = ds.output()
        context = H * dy.softmax(dy.transpose(A * H) * h)
        # Initialize beam
        beam.append((ds, context, [widst['SOS']], 0.0))
        # Loop
        for i in range(int(min(self.max_len, input_len * 1.5))):
            new_beam = []
            for ds, pc, pw, logprob in beam:
                embs = dy.lookup(self.MT_p, pw[-1])
                # Run LSTM
                ds = ds.add_input(dy.concatenate([embs, pc]))
                # Compute next context
                h = ds.output()
                context = H * dy.softmax(dy.transpose(A * H) * h)
                # Get distribution over words
                s = dy.affine_transform([b,D,dy.concatenate([h, context])])
                p = dy.softmax(s * (1 / T)).npvalue().flatten()
                # Careful of float error
                p = p/p.sum()
                kbest = np.argsort(p)
                for nw in kbest[-beam_size:]:
                    new_beam.append((ds, context, pw + [nw], logprob + np.log(p[nw])))

            beam = sorted(new_beam, key=lambda x: x[-1])[-beam_size:]

            if beam[-1][2][-1] == widst['EOS']:
                break

        if debug:
            elapsed = time.time()-start
            print('Decoding : ', elapsed)

        return beam[-1][2]

    def get_components(self):
        return self.MS_p, self.MT_p, self.D_p, self.enc, self.dec, self.A_p, self.rev_enc, self.b_p

    def restore_components(self, components):
        self.MS_p, self.MT_p, self.D_p, self.enc, self.dec, self.A_p, self.rev_enc, self.b_p = components


if __name__ == '__main__':

    # ===================================================================
    if verbose:
        print('Reading corpora')
    # Read vocabs
    widss, ids2ws = read_dic(args.train_src,max_size=args.src_vocab_size)
    widst, ids2wt = read_dic(args.train_dst,max_size=args.trg_vocab_size)
    # Read training
    trainings_data = read_corpus(args.train_src, widss)
    trainingt_data = read_corpus(args.train_dst, widst)
    # Add UNKs
    unk_idx = widss['UNK']
    unk_idx = widst['UNK']
    # Read validation
    valids_data = read_corpus(args.valid_src, widss)
    validt_data = read_corpus(args.valid_dst, widst)
    # Read test
    tests_data = read_corpus(args.test_src, widss)

    # ===================================================================
    if verbose:
        print('Creating model')
        sys.stdout.flush()
    m = dy.Model()
    model_file = args.model
    if model_file is not None:
        [s2s] = m.load(model_file)
    else:
        s2s = Seq2SeqModel(m,
                           args.emb_dim,
                           args.hidden_dim,
                           len(widss),
                           len(widst),
                           bidir=args.bidir,
                           word_emb=args.word_emb,
                           dropout=args.dropout_rate,
                           max_len=args.max_len)
        model_file = args.exp_name+'_model.txt'

    trainer = dy.AdamTrainer(m, args.learning_rate, edecay=args.learning_rate_decay)
    # trainer.set_clip_threshold(-1)
    # ===================================================================
    if verbose:
        print_config()
        sys.stdout.flush()

    # ===================================================================
    if verbose:
        print('Creating batch loaders')
        sys.stdout.flush()
    trainbatchloader = BatchLoader(trainings_data, trainingt_data, args.batch_size)
    devbatchloader = BatchLoader(valids_data, validt_data, args.dev_batch_size)

    # ===================================================================
    if args.train:
        if verbose:
            print('starting training')
            sys.stdout.flush()
        train_loss = 0
        processed = 0
        best_dev_loss = np.inf
        i = 0
        for epoch in range(args.num_epochs):
            start = time.time()
            for x, y in trainbatchloader:
                processed += sum(map(len, y))
                loss = s2s.calculate_loss(x, y)
                loss.backward()
                trainer.update()
                train_loss += loss.scalar_value()
                if (i+1) % args.check_train_error_every == 0:
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
                if (i+1) % args.check_valid_error_every == 0:
                    dev_loss = 0
                    dev_processed = 0
                    dev_start = time.time()
                    for x, y in devbatchloader:
                        dev_processed += sum(map(len, y))
                        loss = s2s.calculate_loss(x, y)
                        dev_loss += loss.scalar_value()
                    dev_logloss = dev_loss/dev_processed
                    dev_ppl = np.exp(dev_logloss)
                    dev_elapsed = time.time()-dev_start
                    print("[epoch %d] Dev loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                          (epoch, dev_logloss, dev_ppl, dev_elapsed, dev_processed))
                    if dev_ppl < best_dev_loss:
                        best_dev_loss = dev_ppl
                        print('Best dev error up to date, saving model to', model_file)
                        m.save(model_file, [s2s])
                    sys.stdout.flush()

                if (i+1) % args.test_every == 0:
                    print('Start running on test set, buckle up!')
                    sys.stdout.flush()
                    test_start = time.time()
                    with open(args.test_out, 'w+') as of:
                        for x in tests_data:
                            y = s2s.translate(x, decoding='beam_search', beam_size=args.beam_size)
                            translation = ' '.join([ids2wt[w] for w in y])
                            of.write(translation+'\n')
                    test_elapsed = time.time()-test_start
                    print('Finished running on test set', test_elapsed, 'elapsed.')
                    sys.stdout.flush()
                i = i+1
            trainer.update_epoch()
    # ===================================================================
    if args.test:
        print('Start running on test set, buckle up!')
        sys.stdout.flush()
        test_start = time.time()
        with open(args.test_out, 'w+') as of:
            for x in tests_data:
                y = s2s.translate(x, decoding='beam_search', beam_size=args.beam_size)
                translation = ' '.join([ids2wt[w] for w in y])
                of.write(translation+'\n')
        test_elapsed = time.time()-test_start
        print('Finished running on test set', test_elapsed, 'elapsed.')
        sys.stdout.flush()
