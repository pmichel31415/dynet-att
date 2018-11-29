#!/usr/bin/env python3

import numpy as np
import dynet as dy
import dynn
from dynn.parameter_initialization import UniformInit


class Decoder(dynn.layers.ParametrizedLayer):
    """Base Encoder class"""

    def __init__(self, pc, args, **kwargs):
        super(Decoder, self).__init__(pc, "dec")

    def init(self, H, y, test=True, update=True):
        pass

    def next(self, w, c, test=True, state=None):
        raise NotImplemented()

    def s(self, h, c, e, test=True):
        raise NotImplemented()


class LSTMDecoder(Decoder):
    """Standard LSTM decoder"""

    def __init__(self, pc, args, **kwargs):
        super(LSTMDecoder, self).__init__(pc, args)
        # Store hyperparameters
        self.dic_tgt = kwargs["dic_tgt"]  # dictionary
        self.V = len(self.dic_tgt)
        self.nl = args.decoder_n_layers  # Number of layers
        self.de = args.decoder_embed_dim  # Embedding dim
        self.dc = args.decoder_ctx_dim  # Context dim
        self.dh = args.decoder_hidden_dim  # Hidden dim
        self.dr = args.dropout  # Dropout
        self.mos_k = args.decoder_mos_k  # Mixture of softmax
        self.di = self.de + self.dc
        # Embeddings
        if hasattr(kwargs, "E_p"):
            E_p = self.pc.add_parameters(
                (self.V, self.de),
                init=UniformInit(1 / np.sqrt(self.de))
            )
        else:
            E_p = getattr(kwargs, "E_p")
        self.tgt_embed = dynn.layers.Embeddings(
            self.pc, self.dic_tgt, self.de, params=E_p)
        # Attention
        self.attend =
        # LSTM Encoder
        self.lstm = dynn.layers.StackedLSTM(
            self.pc, self.nl, self.di, self.dh, self.dr, self.dr
        )
        # Linear layer from last encoding to initial state
        self.proj = dynn.layers.Affine(self.pc, self.dh, self.de)
        # Output layer to logits
        if args.tied_decoder_embeddings:
            self.logit = dynn.layers.Affine(self.pc, self.de, self.V, W_p=E_p)
        else:
            self.logit = dynn.layers.Affine(self.pc, self.de, self.V)
        # Mixture layer
        if self.mos_k > 1:
            self.mos = dynn.layers.Affine(self.pc, self.dh, self.mos_k)

    def init(self, H, y, test=True, update=True):
        self.src_embed.init(test=test, update=update)
        self.lstm.init(test=test, update=update)
        self.proj.init(test=test, update=update)
        self.logit.init(test=test, update=update)
        if self.mos_k > 1:
            self.mos.init(test=test, update=update)

    def __call__(self, X, tgt, attn_mask=None):
        # Embed all words (except EOS)
        tgt_embs = [self.sos.batch([0] * tgt.batch_size)]
        tgt_embs.extend(self.tgt_embed_all([w for w in tgt.sequences[:-1]]))
        # Initialize decoder state
        dec_state = self.dec_cell.initial_value(tgt.batch_size)
        # Iterate over target words
        dec_outputs = []
        for x in tgt_embs:
            # Attention query: previous hidden state and current word embedding
            query = dy.concatenate([x, self.dec_cell.get_output(dec_state)])
            # Attend
            ctx, _ = self.attend(query, X, X, mask=attn_mask)
            # Both context and target word embedding will be fed to the decoder
            dec_input = dy.concatenate([x, ctx])
            # Update decoder state
            dec_state = self.dec_cell(dec_input, *dec_state)
            # Save output
            dec_outputs.append(self.dec_cell.get_output(dec_state))
        # Compute logits
        logits = self.project_all(dec_outputs)

        return logits

    def next(self, w, c, test=True, state=None):
        e = dy.pick_batch(self.E, w)
        if not test:
            e = dy.dropout_dim(e, 0, self.wdr)
        x = dy.concatenate([e, c])
        # Run LSTM
        if state is None:
            self.ds = self.ds.add_input(x)
            next_state = self.ds
        else:
            next_state = state.add_input(x)
        h = next_state.output()
        return h, e, next_state

    def s(self, h, c, e, test=True):
        h = dy.concatenate([h, c, e])
        if self.mos_k == 1:
            output = dy.affine_transform([self.bo[0], self.Wo[0], h])
            if not test:
                output = dy.dropout(output, self.dr)
            # Score
            s = dy.affine_transform([self.b, self.E, output])
            return s
        else:
            outputs = [dy.affine_transform([b, w, h])
                       for w, b in zip(self.Wo, self.bo)]
            log_weights = dy.log_softmax(
                dy.affine_transform([self.bm, self.Wm, h]))
            log_weights = dy.reshape(log_weights, (1, self.mos_k), h.dim()[1])
            logprobs = dy.concatenate_cols(
                [dy.log_softmax(dy.affine_transform([self.b, self.E, o])) for o in outputs])
            logprob = dy.logsumexp_dim(log_weights + logprobs, d=1)
            return logprob

    def load_pretrained(self, filename):
        self.lstm.param_collection().populate(
            filename, self.lstm.param_collection().name())
        self.Wp_p.populate(filename, self.pc.name() + '/Wp')
        self.bp_p.populate(filename, self.pc.name() + '/bp')
        self.Wo_p.populate(filename, self.pc.name() + '/Wo')
        self.bo_p.populate(filename, self.pc.name() + '/bo')
        self.E_p.populate(filename, self.pc.name() + '/E')
        self.b_p.populate(filename, self.pc.name() + '/b')


def get_decoder(decoder, nl, di, de, dh, vt, pc, pre_embs=None, dr=0.0, wdr=0.0, mos_k=1):
    if decoder == 'lm':
        return LSTMLMDecoder(nl, di, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'lstm':
        return LSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs, mos_k=mos_k)
    else:
        print('Unknown decoder type "%s", using lstm decoder' % decoder)
        return LSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs, mos_k=mos_k)
