from __future__ import print_function, division

import numpy as np
import dynet as dy

import encoders
import attention
import decoders
import beam

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


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
                 src_dic,
                 trg_dic,
                 model_file=None,
                 enc_type='lstm',
                 att_type='mlp',
                 dec_type='lstm',
                 lang_model=None,
                 label_smoothing=0.0,
                 dropout=0.0,
                 word_dropout=0.0,
                 max_len=60):
        """Constructor

        :param input_dim: Embedding dimension
        :param hidden_dim: Dimension of the recurrent layers
        :param att_dim: Dimension of the hidden layer in the attention MLP
        :param src_dic: Dictionary of the source language mapping words to indices
        :param trg_dic: Dictionary of the target language mapping words to indices
        :param enc_type: Type of encoder
        :param att_type: Type of attention mechanism
        :param dec_type: Type of decoder
        :param model_file: File where the model should be saved (default: (None))
        :param label_smoothing: interpolation coefficient with second output distribution
        :param dropout: dropout rate for parameters
        :param word_dropout: dropout rate for words in the decoder
        :param max_len: Maximum length allowed when generating translations (default: (60))
        """
        # Store config
        self.nl = num_layers
        self.dr, self.wdr = dropout, word_dropout
        self.ls, self.ls_eps = (label_smoothing > 0), label_smoothing
        self.max_len = max_len
        self.src_sos, self.src_eos = src_dic['SOS'], src_dic['EOS']
        self.trg_sos, self.trg_eos = trg_dic['SOS'], trg_dic['EOS']
        # Dimensions
        self.vs, self.vt = len(src_dic), len(trg_dic)
        self.di, self.dh, self.da = input_dim, hidden_dim, att_dim
        # Model
        self.pc = dy.ParameterCollection('s2s')
        self.model_file = model_file
        # Encoder
        self.enc = encoders.get_encoder(enc_type, self.nl, self.di, self.dh, self.vs, self.pc, dr=self.dr, pre_embs=None)
        # Attention module
        self.att = attention.get_attention(att_type, self.di, self.dh, self.da, self.pc)
        # Decoder
        self.dec = decoders.get_decoder(dec_type, self.nl, self.di, self.enc.dim, self.dh, self.vt, self.pc,
                                        pre_embs=None, dr=self.dr, wdr=self.wdr)

        # Target language model (for label smoothing)
        self.lm = lang_model

        self.test = True
        self.update = True

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

    def encode(self, src):
        """Encode a batch of sentences

        :param src: List of sentences. It is assumed that all
            source sentences have the same length

        :returns: Expression of the encodings
        """
        # Prepare batch
        x, _ = self.prepare_batch(src, self.src_eos)
        self.enc.init(x, test=self.test, update=self.update)
        return self.enc(x, test=self.test)

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
        if self.ls:
            log_prob = dy.log_softmax(s)
            if self.lm is None:
                loss = - dy.pick_batch(log_prob, nw) * (1 - self.ls_eps) - \
                    dy.mean_elems(log_prob) * self.ls_eps
            else:
                loss = - dy.pick_batch(log_prob, nw) * (1 - self.ls_eps) - \
                    dy.dot_product(self.lm.p_next_expr(cw), log_prob) * self.ls_eps
        else:
            loss = dy.pickneglogsoftmax(s, nw)
        return loss

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
        c = dy.zeroes((self.enc.dim,), batch_size=bsize)
        # Start decoding
        errs = []
        for cw, nw, mask in zip(y, y[1:], masksy[1:]):
            # Run LSTM
            h, e, _ = self.dec.next(cw, context, test=self.test)
            # Compute next context
            context, _ = self.attend(encodings, h)
            # Score
            s = self.dec.s(h, context, test=self.test)
            masksy_e = dy.inputTensor(mask, batched=True)
            # Loss
            loss = self.cross_entropy_loss(s, nw, cw)
            loss = dy.cmult(loss, masksy_e)
            errs.append(loss)
        # Add all losses together
        err = dy.mean_batches(dy.esum(errs))
        return err

    def calculate_loss(self, src, trg, test=False):
        """Compute the conditional log likelihood of the target sentences given the source sentences

        Combines encoding and decoding

        :param src: List of sentences. It is assumed that all
                    source sentences have the same length
        :param trg: List of target sentences

        :returns: Expression of the loss averaged on the minibatch
        """
        dy.renew_cg()
        self.lm.init()
        encodings = self.encode(src, test=test)
        err = self.decode_loss(encodings, trg, test=test)
        return err

    def translate(self, x, beam_size=1):
        """Translate a source sentence

        Translate a single source sentence by decoding using beam search

        :param x: Source sentence (list of indices)
        :param beam_size: Size of the beam for beam search.
            A value of 1 means greedy decoding (default: (1))

        :returns generated translation (list of indices)
        """
        dy.renew_cg()
        input_len = len(x)
        encodings = self.encode([x])
        # Decode
        return self.beam_decode(encodings, beam_size)

    def beam_decode(self, encodings, beam_size=1)
        # Add parameters to the graph
        self.dec.init(encodings, [[]], test=self.test, update=self.update)
        # Initialize context
        context = dy.zeroes((self.enc.dim,))
        # Initialize beam
        beams = [beam.Beam(ds, context, [self.trg_sos], 0.0)]
        # Loop
        for i in range(int(min(self.max_len, input_len * 1.5))):
            new_beam = []
            p_list = []
            for b in beams:
                h, e, state = self.dec.next(b.words[-1], b.context, state=b.state)
                # Compute next context
                context, _ = self.attend(encodings, h)
                # Score
                s = self.dec.s(h, context, test=self.test)
                # Probabilities
                p_list.append(dy.softmax(s))
            # Run one forward pass for all elements (maybe leverage autobatching)
            p_list = dy.concatenate_cols(p_list).npvalue().T
            # Only keep the best for each beam
            for p, b in zip(p_list, beam):
                # Careful for floating errors
                p = p.flatten() / p.sum()
                kbest = np.argsort(p)
                for nw in kbest[-beam_size:]:
                    new_beam.append(Beam(b.state, b.context, b.words + [nw], b.logprob + np.log(p[nw])))
            # Only keep the best
            beam = sorted(new_beam, key=lambda b: b.logprob)[-beam_size:]

            if beam[-1].words[-1] == self.trg_eos:
                break

        return beam[-1].words

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
