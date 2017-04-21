from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import dynet as dy


class Seq2SeqModel(object):
    """A neural sequence to sequence model with attention
    
    Uses LSTM encoder and decoder, as well as tanh based attention
    
    Extends:
        object
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 att_dim,
                 src_dic,
                 trg_dic,
                 model_file=None,
                 bidir=False,
                 word_emb=False,
                 dropout=0.0,
                 max_len=60):
        """Constructor
        
        Arguments:
            input_dim (int): Embedding dimension
            hidden_dim (int): Dimension of the recurrent layers
            att_dim (int): Dimension of the hidden layer in the attention MLP
            src_dic (dict): Dictionary of the source language mapping words to indices
            trg_dic (dict): Dictionary of the target language mapping words to indices
        
        Keyword Arguments:
            model_file (str): File where the model should be saved (default: (None))
            bidir (bool): Whether to use a bidirectionnal encoder (default: (False))
            word_emb (bool): Whether to use residual connections between source embeddings and the attention layer (default: (False))
            dropout (number): dropout rate for the output layer (before softmax) (default: (0.0))
            max_len (number): Maximum length allowed when generating translations (default: (60))
        """
        # Store config
        self.bidir = bidir
        self.word_emb = word_emb
        self.dr = dropout
        self.max_len = max_len
        self.src_sos, self.src_eos = src_dic['SOS'], src_dic['EOS']
        self.trg_sos, self.trg_eos = trg_dic['SOS'], trg_dic['EOS']
        # Dimensions
        self.vs, self.vt = len(src_dic), len(trg_dic)
        self.di, self.dh, self.da = input_dim, hidden_dim, att_dim
        self.enc_dim = self.dh
        if self.bidir:
            self.enc_dim += self.dh
        if self.word_emb:
            self.enc_dim += self.di
        self.dec_dim = self.di + self.enc_dim
        self.out_dim = self.di + self.dh+self.enc_dim
        # Model
        self.model = dy.Model()
        self.model_file = model_file
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
        self.Wha_p = self.model.add_parameters((self.da, self.dh))
        # Embedding parameters
        self.MS_p = self.model.add_lookup_parameters((self.vs, self.di))
        self.MT_p = self.model.add_lookup_parameters((self.vt, self.di))
        # Output parameters
        self.Wo_p = self.model.add_parameters((self.di, self.out_dim))
        self.bo_p = self.model.add_parameters((self.di,), init=dy.ConstInitializer(0))
        # Softmax parameters
        self.b_p = self.model.add_parameters((self.vt,), init=dy.ConstInitializer(0))

    def prepare_batch(self, batch, eos):
        """Prepare batch of sentences for sequential processing
        
        Basically transposes the batch, pads sentences of different lengths with EOS symbols and builds a mask for the loss function (so that the loss is masked on the padding words).

        Example (with strings instead of int for clarity):

        [["I","like","chocolate"],["Me","too"]] 
        -> [["I","Me"],["like","too"],["chocolate","EOS"]], [[1,1],[1,1],[1,0]]
        
        Arguments:
            batch (list): List of sentences
            eos (type): EOS index
        
        Returns:
            tuple -- (prepared_batch, masks) both of shape (sentence_length, batch_size)
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

    def encode(self, src, test=False):
        """Encode a batch of sentences
        
        Arguments:
            src (list): List of sentences. It is assumed that all source sentences have the same length
        
        Keyword Arguments:
            test (bool) -- Switch used for things like dropout where the behaviour is different at test time (default: (False)
        
        Returns:
            dynet.Expression -- Expression of the encodings
        """
        # Prepare batch
        x, _ = self.prepare_batch(src, self.src_eos)
        # Add encoder to computation graph
        es = self.enc.initial_state()
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

    def attend(self, encodings, h):
        """Compute attention score
        
        Given :math:`z_i` the encoder's output at time :math:`i`, :math:`h_{j-1}` the decoder's output at time :math:`j-1`, the attention score is computed as :

        .. math::
            
            \begin{split}
                s_{ij}&=V_a^T\tanh(W_az_i + W_{ha}h_j + b_a)\\
                \alpha_{ij}&=\frac{s_{ij}}{\sum_{i'}s_{i'j}}\\
            \end{split}
        
        Arguments:
            encodings (dynet.Expression): Source sentence encodings obtained with self.encode
            h (dynet.Expression): Decoder output at the previous timestep
        
        Returns:
            tuple: Two dynet Expressions, the context and the attention weights
        """
        Va, Wa, Wha = self.Va_p.expr(), self.Wa_p.expr(), self.Wha_p.expr()
        d = dy.tanh(dy.colwise_add(Wa * encodings, Wha * h))
        scores = dy.transpose(d) * Va
        weights = dy.softmax(scores)
        context = encodings * weights
        return context, weights

    def decode_loss(self, encodings, trg, test=False):
        """Compute the negative conditional log likelihood of the target sentence given the encoding of the source sentence
        
        Arguments:
            encodings (dynet.Expression): Source sentence encodings obtained with self.encode
            trg (list): List of target sentences
        
        Keyword Arguments:
            test (bool): Switch used for things like dropout where the behaviour is different at test time (default: (False)
        
        Returns:
            dynet.Expression: Expression of the loss averaged on the minibatch
        """
        y, masksy = self.prepare_batch(trg, self.trg_eos)
        slen, bsize = y.shape
        # Add parameters to the graph
        Wp, bp = self.Wp_p.expr(), self.bp_p.expr()
        Wo, bo = self.Wo_p.expr(), self.bo_p.expr()
        D, b = dy.transpose(dy.parameter(self.MT_p)), self.b_p.expr()
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
            context, _ = self.attend(encodings, h)
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
        """Compute the conditional log likelihood of the target sentences given the source sentences
        
        Combines encoding and decoding
        
        Arguments:
            src (list): List of sentences. It is assumed that all source sentences have the same length
            trg (list): List of target sentences
        
        Keyword Arguments:
            test (bool): Switch used for things like dropout where the behaviour is different at test time (default: (False)
        
        Returns:
            dynet.Expression: Expression of the loss averaged on the minibatch
        """
        dy.renew_cg()
        encodings = self.encode(src, test=test)
        err = self.decode_loss(encodings, trg, test=test)
        return err

    def translate(self, x, beam_size=1):
        """Translate a source sentence
        
        Translate a single source sentence by decoding using beam search

        Arguments:
            x (list): Source sentence (list of indices)
        
        Keyword Arguments:
            beam_size (int): Size of the beam for beam search. A value of 1 means greedy decoding (default: (1))
        
        Returns:
            list: generated translation (list of indices)
        """
        dy.renew_cg()
        input_len = len(x)
        encodings = self.encode([x], test=True)
        # Decode
        # Add parameters to the graph
        Wp, bp = self.Wp_p.expr(), self.bp_p.expr()
        Wo, bo = self.Wo_p.expr(), self.bo_p.expr()
        D, b = dy.transpose(dy.parameter(self.MT_p)), self.b_p.expr()
        # Initialize decoder with last encoding
        last_enc = dy.select_cols(encodings, [encodings.dim()[0][-1] - 1])
        init_state = dy.affine_transform([bp, Wp, last_enc])
        ds = self.dec.initial_state([init_state, dy.zeroes((self.dh,))])
        # Initialize context
        context = dy.zeroes((self.enc_dim,))
        # Initialize beam
        beam = [(ds, context, [self.trg_sos], 0.0)]
        # Loop
        for i in range(int(min(self.max_len, input_len * 1.5))):
            new_beam = []
            for ds, pc, pw, logprob in beam:
                embs = dy.lookup(self.MT_p, pw[-1])
                # Run LSTM
                ds = ds.add_input(dy.concatenate([embs, pc]))
                h = ds.output()
                # Compute next context
                context, _ = self.attend(encodings, h)
                # Compute output with residual connections
                output = dy.affine_transform([bo, Wo, dy.concatenate([h, context, embs])])
                # Score
                s = dy.affine_transform([b, D, output])
                # Probabilities
                p = dy.softmax(s).npvalue().flatten()
                # Careful of float error
                p = p / p.sum()
                kbest = np.argsort(p)
                for nw in kbest[-beam_size:]:
                    new_beam.append((ds, context, pw + [nw], logprob + np.log(p[nw])))

            beam = sorted(new_beam, key=lambda x: x[-1])[-beam_size:]

            if beam[-1][2][-1] == self.trg_eos:
                break

        return beam[-1][2]

    def save(self):
        """Save model
        
        Saves the model holding the parameters to self.model_file
        """
        self.model.save(self.model_file)

    def load(self):
        """Load model
        
        Loads the model holding the parameters from self.model_file
        """
        self.model.load(self.model_file)
