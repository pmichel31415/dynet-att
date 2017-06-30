from __future__ import print_function, division

import numpy as np
import dynet as dy

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


class Decoder(object):
    """Base Encoder class"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('dec')

    def init(self):
        pass

    def __call__(self, x):
        raise NotImplemented()


class LSTMLMDecoder(Encoder):
    """docstring for EmptyEncoder"""

    def __init__(self, nl, di, dh, vt, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(LSTMLMDecoder, self).__init__(pc)

        # Store hyperparameters
        self.nl, self.di, self.dh = nl, di, dh
        self.dr, self.wdr = dr, wdr
        self.vt = vt
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)
        # Output layer
        self.Wo_p = self.model.add_parameters((self.di, self.dh + self.di), name='Wo')
        self.bo_p = self.model.add_parameters((self.di,), name='bo')
        # Embedding matrix
        self.E = self.pc.add_parameters((self.vs, self.di), name='E')
        if pre_embs is not None:
            self.E.set_value(pre_embs)

    def init(self, H, y, test=True, update=True):
        bs = len(y[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Add encoder to computation graph
        self.ds = self.lstm.initial_state(update=update)
        if not test:
            self.lstm.set_dropout_masks(bs)

        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.E = self.E_p.expr(update)

    def next(self, w, c, test=True, state=None):
        e = dy.pick_batch(D, w)
        if not test:
            e = dy.dropout_dim(e, 0, self.wdr)
        # Run LSTM
        if state is None:
            self.ds = self.ds.add_input(e)
            next_state = self.ds
        else:
            next_state = state.add_input(e)
        h = ds.output()
        return h, e, next_state

    def s(self, h, c, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        s = D * output
        return s


class LSTMDecoder(Decoder):
    """docstring for LSTMDecoder"""

    def __init__(self, nl, di, de, dh, vt, pc, pre_embs=None, dr=0.0, wdr=0.0):
        super(LSTMDecoder, self).__init__(pc)
        # Store hyperparameters
        self.nl, self.di, self.de, self.dh = nl, di, de, dh
        self.dr, self.wdr = dr, wdr
        self.vt = vt
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di + self.de, self.dh, self.pc)
        # Linear layer from last encoding to initial state
        self.Wp_p = self.model.add_parameters((self.di, self.de), name='Wp')
        self.bp_p = self.model.add_parameters((self.di,), name='bp')
        # Output layer
        self.Wo_p = self.model.add_parameters((self.di, self.dh + self.de + self.di), name='Wo')
        self.bo_p = self.model.add_parameters((self.di,), name='bo')
        # Embedding matrix
        self.E = self.pc.add_parameters((self.vs, self.di), name='E')
        if pre_embs is not None:
            self.E.set_value(pre_embs)

    def init(self, H, y, test=True, update=True):
        bs = len(y[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Initialize first state of the decoder with the last state of the encoder
        self.Wp = self.Wp_p.expr(update)
        self.bp = self.bp_p.expr(update)
        last_enc = dy.pick(H, index=H.dim()[0][-1] - 1, dim=1)
        init_state = dy.affine_transform([self.bp, self.Wp, last_enc])
        ds = self.dec.initial_state([init_state, dy.zeroes((self.dh,), batch_size=bs)], update=update)
        # Initialize dropout masks
        if not test:
            self.lstm.set_dropout_masks(bs)

        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.E = self.E_p.expr(update)

    def next(self, w, c, test=True, state=None):
        e = dy.pick_batch(D, w)
        if not test:
            e = dy.dropout_dim(e, 0, self.wdr)
        x = dy.concatenate([e, c])
        # Run LSTM
        if state is None:
            self.ds = self.ds.add_input(x)
            next_state = self.ds
        else:
            next_state = state.add_input(x)
        h = ds.output()
        return h, e, next_state

    def s(self, h, c, test=True):
        output = dy.affine_transform([self.bo, self.Wo, dy.concatenate([h, c, e])])
        if not test:
            output = dy.dropout(output, self.dr)
        # Score
        s = D * output
        return s


def get_decoder(decoder, nl, di, de, dh, vt, pc, pre_embs=None, dr=0.0, wdr=0.0):
    if decoder == 'lm':
        return LSTMLMDecoder(nl, di, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    elif decoder == 'lstm':
        return LSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
    else:
        print('Unknown decoder type "%s", using lstm decoder' % decoder)
        return LSTMDecoder(nl, di, de, dh, vt, pc, dr=dr, wdr=wdr, pre_embs=pre_embs)
