#!/usr/bin/env python3

import numpy as np
import dynet as dy

import encoders
import attention
import decoders
import beam
import loss



class Seq2SeqModel(object):
    """A neural sequence to sequence model with attention

    Uses LSTM encoder and decoder, as well as tanh based attention

    Extends:
        object
    """

    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 att_dim,
                 vocab,
                 model_file=None,
                 enc_type='lstm',
                 att_type='mlp',
                 dec_type='lstm',
                 loss_type='cross_entropy',
                 pretrained_wembs=None,
                 label_smoothing=0.0,
                 kbest_synonyms=0,
                 dropout=0.0,
                 word_dropout=0.0,
                 unk_replacement=False,
                 log_unigram_bias=False,
                 mos_k=1,
                 desentencepiece=False,
                 max_len=60):
        """Constructor"""
        # Store config
        self.nl = num_layers
        self.dr, self.wdr = dropout, word_dropout
        self.ls, self.ls_eps = (label_smoothing > 0), label_smoothing
        self.kbest_synonyms = kbest_synonyms
        self.max_len = max_len
        self.src_sos, self.src_eos = vocab.w2ids['SOS'], vocab.w2ids['EOS']
        self.trg_sos, self.trg_eos = vocab.w2idt['SOS'], vocab.w2idt['EOS']
        # Dimensions
        self.vs, self.vt = len(vocab.w2ids), len(vocab.w2idt)
        self.di, self.dh, self.da = input_dim, hidden_dim, att_dim
        # Model
        self.pc = dy.ParameterCollection()
        self.model_file = model_file
        # Encoder
        self.enc = encoders.get_encoder(enc_type, self.nl, self.di,
                                        self.dh, self.vs, self.pc,
                                        dr=self.dr, pre_embs=pretrained_wembs)
        # Attention module
        self.att = attention.get_attention(att_type, self.enc.dim, self.dh, self.da, self.pc)
        # Decoder
        self.dec = decoders.get_decoder(dec_type, self.nl, self.di,
                                        self.enc.dim, self.dh, self.vt,
                                        self.pc, pre_embs=pretrained_wembs, dr=self.dr, wdr=self.wdr, mos_k=mos_k)
        self.loss_function = loss.loss_functions[loss_type](epsilon=self.ls_eps,
                                                           vocab=vocab, kbest=kbest_synonyms)
        self.loss_type=loss_type
        # Target language model (for label smoothing)

        self.vocab = vocab
        self.unk_replace = unk_replacement
        self.test = True
        self.update = True
        self.unigram_interpolation = log_unigram_bias
        self.desentencepiece = desentencepiece

    def set_test_mode(self):
        self.test = True

    def set_train_mode(self):
        self.test = False

    def freeze_parameters(self):
        self.update = False

    def thaw_parameters(self):
        self.update = True

    def prepare_batch(self, batch, eos):
        """Prepare batch of sentences for sequential processing

        Basically transposes the batch, pads sentences of different lengths
            with EOS symbols and builds a mask for the loss function
            (so that the loss is masked on the padding words).

        Example (with strings instead of int for clarity):

        [["I","like","chocolate"],["Me","too"]]
        -> [["I","Me"],["like","too"],["chocolate","EOS"]], [[1,1],[1,1],[1,0]]

        :param batch: List of sentences
        :param eos: EOS index

        :returns: (prepared_batch, masks) both of shape (sentence_length, batch_size)
        """
        bsize = len(batch)

        batch_len = max(len(s) for s in batch)

        x = np.zeros((batch_len, bsize), dtype=int)
        masks = np.ones((batch_len, bsize), dtype=float)
        x[:] = eos

        for i in range(bsize):
            sent = batch[i][:]
            masks[len(sent):, i] = 0.0
            while len(sent) < batch_len:
                sent.append(eos)
            x[:, i] = sent
        return x, masks

    def label_noise(self, batch):
        if self.ls:
            for i, y in enumerate(batch):
                threshold = np.random.uniform(0,1, size=len(y))
                noisy_labels = np.random.randint(len(self.vocab.id2wt), size=len(y))
                batch[i] = [noisy_labels[k] if threshold[k] < self.ls_eps else y[k] for k in range(len(y))]
        return batch

    def encode(self, src, test=False):
        """Encode a batch of sentences

        :param src: List of sentences. It is assumed that all
            source sentences have the same length

        :returns: Expression of the encodings
        """
        # Prepare batch
        x, _ = self.prepare_batch(src, self.src_eos)
        self.enc.init(x, test=self.test, update=self.update)
        return self.enc(x, test=self.test, update=self.update)

    def attend(self, encodings, h):
        """Compute attention score

        Given :math:`z_i` the encoder's output at time :math:`i`, :math:`h_{j-1}`
        the decoder's output at time :math:`j-1`, the attention score is computed as :

        .. math::

            \begin{split}
                s_{ij}&=V_a^T\tanh(W_az_i + W_{ha}h_j + b_a)\\
                \alpha_{ij}&=\frac{s_{ij}}{\sum_{i'}s_{i'j}}\\
            \end{split}

        :param encodings: Source sentence encodings obtained with self.encode
        :param h: Decoder output at the previous timestep

        :returns: Two dynet Expressions, the context and the attention weights
        """
        self.att.init(test=self.test, update=self.update)
        return self.att(encodings, h, test=self.test)

    def cross_entropy_loss(self, s, nw, cw):
        """Calculates the cross-entropy
        """
        if self.unigram_interpolation:
            s = s + dy.log(self.lm.p_next_expr(cw) + self.ls_eps)
        log_prob = dy.log_softmax(s)
        nll = - dy.pick_batch(log_prob, nw)
        if self.ls and not self.unigram_interpolation:
            if self.lm is None:
                if self.kbest_synonyms>0:
                    ls_losses = []
                    synonyms = self.vocab.kbest[nw]
                    masks = np.zeros((self.kbest_synonyms + 1, len(synonyms)))
                    syns_arr = np.zeros((self.kbest_synonyms + 1 , len(synonyms)), dtype=int)
                    for i, syns in enumerate(synonyms):
                        w_and_syns = [nw[i]] + syns[:self.kbest_synonyms]
                        masks[:len(w_and_syns), i] = 1 / len(w_and_syns)
                        syns_arr[:len(w_and_syns), i] = w_and_syns
                    for syn, mask in zip(syns_arr, masks):
                        mask_e = dy.inputTensor(mask, batched=True)
                        ls_losses.append(dy.cmult(dy.pick_batch(log_prob, syn, 0), mask_e))
                    ls_loss = - dy.esum(ls_losses)
                else:
                    ls_loss = dy.mean_elems(dy.square(s))
            else:
                ls_loss = - dy.dot_product(self.lm.p_next_expr(cw), log_prob)
        else:
            ls_loss = dy.zeros(1)
        return nll, ls_loss

    def decode_loss(self, encodings, trg):
        """Compute the negative conditional log likelihood of the target sentence
        given the encoding of the source sentence

        :param encodings: Source sentence encodings obtained with self.encode
        :param trg: List of target sentences

        :returns: Expression of the loss averaged on the minibatch
        """
        y, masksy = self.prepare_batch(trg, self.trg_eos)
        slen, bsize = y.shape
        # Init decoder
        self.dec.init(encodings, y, test=self.test, update=self.update)
        # Initialize context
        context = dy.zeroes((self.enc.dim,), batch_size=bsize)
        # Start decoding
        nlls, ls_losses, losses = [], [], []
        for cw, nw, mask in zip(y, y[1:], masksy[1:]):
            # Run LSTM
            h, e, _ = self.dec.next(cw, context, test=self.test)
            # Compute next context
            context, _ = self.attend(encodings, h)
            # Score
            s = self.dec.s(h, context, e, test=self.test)
            # Loss
            loss_obj = self.loss_function(s, nw)
            masksy_e = dy.inputTensor(mask, batched=True)
            nll = dy.cmult(loss_obj.nll, masksy_e)
            ls_loss = dy.cmult(loss_obj.ls_loss, masksy_e)
            full_loss = dy.cmult(loss_obj.loss, masksy_e)
            nlls.append(nll)
            ls_losses.append(ls_loss)
            losses.append(full_loss)
        # Add all losses together
        tot_nll = dy.mean_batches(dy.esum(nlls))
        tot_ls_loss = dy.mean_batches(dy.esum(ls_losses))
        tot_loss = dy.mean_batches(dy.esum(losses))
        return loss.Loss(tot_nll, tot_ls_loss, tot_loss)

    def calculate_loss(self, src, trg, test=False):
        """Compute the conditional log likelihood of the target sentences given the source sentences

        Combines encoding and decoding

        :param src: List of sentences. It is assumed that all
                    source sentences have the same length
        :param trg: List of target sentences

        :returns: Expression of the loss averaged on the minibatch
        """
        dy.renew_cg()
        self.loss_function.init()
        encodings = self.encode(src)
        #if not test:
        #    self.label_noise(trg)
        err = self.decode_loss(encodings, trg)
        return err
   
    def join_words(self, sent):
        if self.desentencepiece:
            string = ''.join(sent).replace('\u2581', ' ').strip()
        else:
            string = ' '.join(sent)
        return string

    def translate(self, src, beam_size=1, kbest=1):
        """Translate a source sentence

        Translate a single source sentence by decoding using beam search

        :param src: Source sentence (list of strings)
        :param beam_size: Size of the beam for beam search.
            A value of 1 means greedy decoding (default: (1))

        :returns generated translation (list of indices)
        """
        dy.renew_cg()
        x = self.vocab.sent_to_ids(src)
        input_len = len(x)
        encodings = self.encode([x], test=True)
        # Decode
        beams = self.beam_decode(encodings, input_len=len(x), beam_size=beam_size, k=kbest)
        # Post process (unk replacement...)
        sents = [self.post_process(b, src) for b in beams]
        if kbest == 1:
            return self.join_words(sents[-1])
        else:
            return '\t'.join(self.join_words(sent) for sent in sents)

    def beam_decode(self, encodings, input_len=10, beam_size=1, k=1):
        # Add parameters to the graph
        self.dec.init(encodings, [[self.trg_sos]],
                      test=self.test, update=self.update)
        # Initialize context
        context = dy.zeroes((self.enc.dim,))
        # Get conditional log probability of lengths
        llp = np.log(self.vocab.p_L[input_len]+1e-20)
        # Initialize beam
        beams = [beam.Beam(self.dec.ds, context, [self.trg_sos], llp[1])]
        # Loop
        for i in range(int(min(self.max_len, input_len * 1.5))):
            new_beam = []
            for b in beams:
                if b.words[-1] == self.trg_eos:
                    new_beam.append(beam.Beam(b.state, b.context, b.words, b.logprob, b.align))
                    continue
                h, e, b.state = self.dec.next([b.words[-1]], b.context, state=b.state)
                # Compute next context
                b.context, att = self.attend(encodings, h)
                # Score
                s = self.dec.s(h, b.context, e, test=self.test)
                # Probabilities
                p = dy.softmax(s).npvalue()
                # Careful for floating errors
                p = p.flatten() / p.sum()
                if 'augmenting' in self.loss_type:
                    p = self.ls_eps * self.vocab.trg_unigrams + (1 - self.ls_eps) * p
                # Store alignment for e.g. unk replacement
                align = np.argmax(att.npvalue())
                kbest = np.argsort(p)
                for nw in kbest[-beam_size:]:
                    new_beam.append(beam.Beam(b.state, b.context,
                                              b.words + [nw],
                                              b.logprob + np.log(p[nw]) + llp[i + 2] - llp[i+1],
                                              b.align + [align]))
            # Only keep the best
            beams = sorted(new_beam, key=lambda b: b.logprob)[-beam_size:]
            over = [(b.words[-1] == self.trg_eos) for b in beams[-k:]]
            
            if all(over):
                break

        return beams[-k:]

    def post_process(self, b, src):
        sent = self.vocab.ids_to_sent(b.words, trg=True)
        if self.unk_replace:
            for i, w in enumerate(sent):
                if w == 'UNK':
                    sent[i] = self.vocab.translate(src[b.align[i + 1]])
        return sent

    def save(self):
        """Save model

        Saves the model holding the parameters to self.model_file
        """
        self.pc.save(self.model_file)

    def load(self):
        """Load model

        Loads the model holding the parameters from self.model_file
        """
        self.pc.populate(self.model_file)

