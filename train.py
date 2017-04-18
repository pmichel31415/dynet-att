import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from __future__ import print_function, division
import numpy as np
import argparse
from collections import defaultdict
import dynet as dy
import time
import pickle
from nltk.translate.bleu_score import corpus_bleu

CHECK_TRAIN_ERROR_EVERY = 10
CHECK_VALID_ERROR_EVERY = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--dynet-seed", default=0, type=int)
parser.add_argument("--dynet-mem", default=512, type=int)
parser.add_argument("--dynet-gpus", default=0, type=int)
parser.add_argument("--train_src", '-ts',
                    default='en-de/train.en-de.de', type=str)
parser.add_argument("--train_dst", '-td',
                    default='en-de/train.en-de.en', type=str)
parser.add_argument("--valid_src", '-vs',
                    default='en-de/valid.en-de.de', type=str)
parser.add_argument("--valid_dst", '-vd',
                    default='en-de/valid.en-de.en', type=str)
parser.add_argument("--test_src", '-tes',
                    default='en-de/test.en-de.de', type=str)
parser.add_argument("--test_dst", '-ted',
                    default='en-de/test.en-de.en', type=str)
parser.add_argument("--test_out", '-teo',
                    default='results/out.en-de.en', type=str)
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
parser.add_argument('--att_dim', '-da',
                    type=int, help='attention size', default=256)
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


def read_dic(file, max_size=20000, min_freq=1):
    dic = defaultdict(lambda: 0)
    freqs = defaultdict(lambda: 0)
    dic['UNK'], dic['SOS'], dic['EOS'] = 0, 1, 2
    with open(file, 'r') as f:
        for l in f:
            sent = l.strip().split()
            for word in sent:
                freqs[word] += 1

    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    for i in range(max_size):
        word, freq = sorted_words[i]
        if freq <= min_freq:
            continue
        dic[word] = len(dic)

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
        self.batches = []

        self.bs = bsize

        # Bucket samples by source sentence length
        buckets = defaultdict(list)
        for src, trg in zip(datas, datat):
            buckets[len(src)].append((src, trg))

        for src_len, bucket in buckets.items():
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
                 att_dim,
                 source_vocab_size,
                 target_vocab_size,
                 bidir=False,
                 word_emb=False,
                 dropout=0.0,
                 max_len=60):
        # Store config
        self.bidir = bidir
        self.word_emb = word_emb
        self.dr = dropout
        self.max_len = max_len
        # Dimensions
        self.vs, self.vt = source_vocab_size, target_vocab_size
        self.di, self.dh, self.da = input_dim, hidden_dim, att_dim
        self.enc_dim = self.dh
        if self.bidir:
            self.enc_dim += self.dh
        if self.word_emb:
            self.enc_dim += self.di
        self.dec_dim = self.di + self.enc_dim
        self.out_dim = self.di + self.dh+self.enc_dim
        # Model
        self.model = model
        # RNN parameters
        self.enc = dy.VanillaLSTMBuilder(1, self.di, self.dh, self.model)
        self.rev_enc = dy.VanillaLSTMBuilder(1, self.di, self.dh, self.model)
        self.dec = dy.VanillaLSTMBuilder(1, self.dec_dim, self.dh, self.model)
        # State passing parameters
        self.Wp_p = self.model.add_parameters((self.dh, self.enc_dim))
        self.bp_p = self.model.add_parameters((self.dh,), init=dy.ConstInitializer(0))
        # Attention parameters
        self.Va_p = self.model.add_parameters((self.da))
        self.Wa_p = self.model.add_parameters((self.da, self.enc_dim))
        self.Wha_p = self.model.add_parameters((self.da,self.dh))
        # Embedding parameters
        self.MS_p = self.model.add_lookup_parameters((self.vs, self.di))
        self.MT_p = self.model.add_lookup_parameters((self.vt, self.di))
        # Output parameters
        self.Wo_p = self.model.add_parameters((self.di, self.out_dim))
        self.bo_p = self.model.add_parameters((self.di,), init=dy.ConstInitializer(0))
        # Softmax parameters
        self.D_p = self.model.add_parameters((self.vt, self.di))
        self.b_p = self.model.add_parameters((self.vt,), init=dy.ConstInitializer(0))

    def prepare_batch(self, batch, dic):
        bsize = len(batch)

        batch_len = max(len(s) for s in batch)

        x = np.zeros((batch_len, bsize), dtype=int)
        masks = np.ones((batch_len, bsize), dtype=float)
        x[:] = dic['EOS']

        for i in range(bsize):
            sent = batch[i][:]
            masks[len(sent):, i] = 0.0
            while len(sent) < batch_len:
                sent.append(dic['EOS'])
            x[:, i] = sent
        return x, masks

    def encode(self, src, test=False):
        x, _ = self.prepare_batch(src, widss)
        es = self.enc.initial_state()
        encoded_states = []
        # Embed words
        wembs = [dy.lookup_batch(self.MS_p, iw) for iw in x]
        # Encode sentence
        encoded_states = es.transduce(wembs)
        # Use bidirectional encoder
        if self.bidir:
            res = self.rev_enc.initial_state()
            rev_encoded_states = res.transduce(wembs[::-1])[::-1]
        # Create encoding matrix
        H = dy.concatenate_cols(encoded_states)
        if self.bidir:
            H_bidir = dy.concatenate_cols(rev_encoded_states)
            H = dy.concatenate([H, H_bidir])
        if self.word_emb:
            H_word_embs = dy.concatenate_cols(wembs)
            H = dy.concatenate([H, H_word_embs])

        return H

    def attend(self, encodings, h, embs):
        Va, Wa, Wha = self.Va_p.expr(), self.Wa_p.expr(), self.Wha_p.expr()
        d = dy.tanh(dy.colwise_add(Wa * encodings,Wha * h))
        scores = dy.transpose(d) * Va
        weights = dy.softmax(scores)
        context = encodings * weights
        return context, weights

    def decode_loss(self, encodings, trg, test=False):
        y, masksy = self.prepare_batch(trg, widst)
        slen, bsize = y.shape
        # Add parameters to the graph
        Wp, bp = self.Wp_p.expr(), self.bp_p.expr()
        Wo, bo = self.Wo_p.expr(), self.bo_p.expr()
        D, b = self.D_p.expr(), self.b_p.expr()
        # Initialize decoder with last encoding
        last_enc = dy.select_cols(encodings, [encodings.dim()[0][-1] - 1])
        init_state = dy.affine_transform([bp, Wp, last_enc])
        ds = self.dec.initial_state([init_state, dy.zeroes((self.dh,), batch_size=bsize)])
        # Initialize context
        context = dy.zeroes((self.enc_dim,), batch_size=bsize)
        # Start decoding
        errs = []
        for cw, nw, mask in zip(y, y[1:], masksy[1:]):
            embs = dy.lookup_batch(self.MT_p, cw)
            # Run LSTM
            ds = ds.add_input(dy.concatenate([embs, context]))
            h = ds.output()
            # Compute next context
            context, _ = self.attend(encodings, h, embs)
            # Compute output with residual connections
            output = dy.affine_transform([bo, Wo, dy.concatenate([h, context, embs])])
            if not test:
                output = dy.dropout(output, self.dr)
            # Score
            s = dy.affine_transform([b, D, output])
            masksy_e = dy.inputTensor(mask, batched=True)
            # Loss
            err = dy.cmult(dy.pickneglogsoftmax_batch(s, nw), masksy_e)
            errs.append(err)
        # Add all losses together
        err = dy.sum_batches(dy.esum(errs)) / float(bsize)
        return err

    def calculate_loss(self, src, trg, test=False):
        dy.renew_cg()
        encodings = self.encode(src, test=test)
        err = self.decode_loss(encodings, trg, test=test)
        return err

    def translate(self, x, decoding='greedy', T=1.0, beam_size=1):
        dy.renew_cg()
        input_len = len(x)
        encodings = self.encode([x], test=True)
        # Decode
        # Add parameters to the graph
        Wp, bp = self.Wp_p.expr(), self.bp_p.expr()
        Wo, bo = self.Wo_p.expr(), self.bo_p.expr()
        D, b = self.D_p.expr(), self.b_p.expr()
        # Initialize decoder with last encoding
        last_enc = dy.select_cols(encodings, [encodings.dim()[0][-1] - 1])
        init_state = dy.affine_transform([bp, Wp, last_enc])
        ds = self.dec.initial_state([init_state, dy.zeroes((self.dh,))])
        # Initialize context
        context = dy.zeroes((self.enc_dim,))
        # Initialize beam
        beam = [(ds, context, [widst['SOS']], 0.0)]
        # Loop
        for i in range(int(min(self.max_len, input_len * 1.5))):
            new_beam = []
            for ds, pc, pw, logprob in beam:
                embs = dy.lookup(self.MT_p, pw[-1])
                # Run LSTM
                ds = ds.add_input(dy.concatenate([embs, pc]))
                h=ds.output()
                # Compute next context
                context, _ = self.attend(encodings, h, embs)
                # Compute output with residual connections
                output = dy.affine_transform([bo, Wo, dy.concatenate([h, context, embs])])
                # Score
                s = dy.affine_transform([b, D, output])
                # Probabilities
                p = dy.softmax(s * (1 / T)).npvalue().flatten()
                # Careful of float error
                p = p / p.sum()
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

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model.load(file)


if __name__ == '__main__':

    # ===================================================================
    if verbose:
        print('Reading corpora')
    # Read vocabs
    widss, ids2ws = read_dic(args.train_src, max_size=args.src_vocab_size)
    widst, ids2wt = read_dic(args.train_dst, max_size=args.trg_vocab_size)
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
    testt_data = read_corpus(args.test_dst, widst)

    # ===================================================================
    if verbose:
        print('Creating model')
        sys.stdout.flush()
    m = dy.Model()
    model_file = args.model
    s2s = Seq2SeqModel(m,
                       args.emb_dim,
                       args.hidden_dim,
                       args.att_dim,
                       len(widss),
                       len(widst),
                       bidir=args.bidir,
                       word_emb=args.word_emb,
                       dropout=args.dropout_rate,
                       max_len=args.max_len)

    if model_file is not None:
        s2s.load(model_file)
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
        start = time.time()
        train_loss = 0
        processed = 0
        best_dev_loss = np.inf
        i = 0
        for epoch in range(args.num_epochs):
            for x, y in trainbatchloader:
                processed += sum(map(len, y))
                bsize = len(y)
                loss = s2s.calculate_loss(x, y)
                loss.backward()
                trainer.update()
                train_loss += loss.scalar_value() * bsize
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
                        print('Best dev error up to date, saving model to', model_file)
                        s2s.save(model_file)
                    sys.stdout.flush()
                    start = time.time()

                if (i+1) % args.test_every == 0:
                    print('Start running on test set, buckle up!')
                    sys.stdout.flush()
                    test_start = time.time()
                    translations=[]
                    references=[]
                    empty=False
                    for x,y in zip(tests_data,testt_data):
                        y_hat = s2s.translate(x, decoding='beam_search', beam_size=args.beam_size)
                        reference = [ids2wt[w] for w in y[1:-1]]
                        translation = [ids2wt[w] for w in y_hat[1:-1]]
                        print('##### REFERENCE #####')
			print(' '.join(reference))
                        print('#### TRANSLATION ####')
			print(' '.join(translation))
                        if len(translation)<1:
                            empty=True
                            break
                        references.append([reference])
                        translations.append(translation)
                    test_elapsed = time.time()-test_start
                    if empty:
                        bleu=0
                    else:
                        bleu = corpus_bleu(references, translations)*100
                    print('Finished running on test set', test_elapsed, 'elapsed, BLEU score :',bleu)
                    sys.stdout.flush()
                    start = time.time()
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
