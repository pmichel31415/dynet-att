from __future__ import print_function, division

import dynet as dy

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


class Encoder(object):
    """Base Encoder class"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('enc')
        self.dim = 0

    def init(self, x, test=True, update=True):
        pass

    def __call__(self, x, test=True):
        raise NotImplemented()


class EmptyEncoder(Encoder):
    """docstring for EmptyEncoder"""

    def __init__(self, pc):
        super(EmptyEncoder, self).__init__(pc)

    def __call__(self, x, test=True):
        return 0


class LSTMEncoder(Encoder):
    """docstring for LSTMEncoder"""

    def __init__(self, nl, di, dh, vs, pc, dr=0.0, pre_embs=None):
        super(LSTMEncoder, self).__init__(pc)
        # Store hyperparameters
        self.nl, self.di, self.dh = nl, di, dh
        self.dr = dr
        self.vs = vs
        self.dim += dh
        # LSTM Encoder
        self.lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)
        # Embedding matrix
        self.E = self.pc.add_lookup_parameters((self.vs, self.di), name='E')
        if pre_embs is not None:
            self.E.init_from_array(pre_embs)

    def init(self, x, test=True, update=True):
        bs = len(x[0])
        if not test:
            self.lstm.set_dropout(self.dr)
        else:
            self.lstm.disable_dropout()
        # Add encoder to computation graph
        self.es = self.lstm.initial_state(update=update)
        if not test:
            self.lstm.set_dropout_masks(bs)

    def __call__(self, x, test=True):
        wembs = [dy.lookup_batch(self.E, iw) for iw in x]
        # Encode sentence
        encoded_states = self.es.transduce(wembs)
        # Create encoding matrix
        H = dy.concatenate_cols(encoded_states)
        return H


class BiLSTMEncoder(LSTMEncoder):
    """docstring for BiLSTMEncoder"""

    def __init__(self, nl, di, dh, vs, pc, dr=0.0, pre_embs=None):
        super(BiLSTMEncoder, self).__init__(nl, di, dh, vs, pc, dr, pre_embs)
        self.dim += dh
        # Backward encoder
        self.rev_lstm = dy.VanillaLSTMBuilder(self.nl, self.di, self.dh, self.pc)

    def init(self, x, test=True, update=True):
        super(BiLSTMEncoder, self).init(x, test, update)
        bs = len(x[0])
        if not test:
            self.rev_lstm.set_dropout(self.dr)
        else:
            self.rev_lstm.disable_dropout()
        # Add encoder to computation graph
        self.res = self.rev_lstm.initial_state(update=update)

        if not test:
            self.rev_lstm.set_dropout_masks(bs)

    def __call__(self, x, test=True):
        # Embed words
        wembs = [dy.lookup_batch(self.E, iw) for iw in x]
        # Encode sentence
        encoded_states = self.es.transduce(wembs)
        rev_encoded_states = self.res.transduce(wembs[::-1])[::-1]
        # Create encoding matrix
        H_fwd = dy.concatenate_cols(encoded_states)
        H_bwd = dy.concatenate_cols(rev_encoded_states)
        H = dy.concatenate([H_fwd, H_bwd])

        return H


def get_encoder(encoder, nl, di, dh, vs, pc, dr=0.0, pre_embs=None):
    if encoder == 'empty':
        return EmptyEncoder(pc)
    elif encoder == 'lstm':
        return LSTMEncoder(nl, di, dh, vs, pc, dr=dr, pre_embs=pre_embs)
    elif encoder == 'bilstm':
        return BiLSTMEncoder(nl, di, dh, vs, pc, dr=dr, pre_embs=pre_embs)
    else:
        print('Unknown encoder type "%s", using bilstm encoder' % encoder)
        return BiLSTMEncoder(nl, di, dh, vs, pc, dr=dr, pre_embs=pre_embs)
