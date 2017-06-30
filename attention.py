from __future__ import print_function, division

import dynet as dy

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


class Attention(object):
    """docstring for Attention"""

    def __init__(self, pc):
        self.pc = pc.add_subcollection('att')

    def init(self, test=True, update=True):
        pass

    def __call__(self, H, h, test=True):
        raise NotImplemented()


class EmptyAttention(Attention):
    """docstring for EmptyAttention"""

    def __init__(self, pc):
        super(EmptyAttention, self).__init__(pc)

    def __call__(self, H, h, test=True):
        return 0, 0


class MLPAttention(Attention):
    """docstring for MLPAttention"""

    def __init__(self, di, dh, da, pc):
        super(MLPAttention, self).__init__(pc)
        self.di, self.dh, self.da = di, dh, da
        # Parameters
        self.Va_p = self.pc.add_parameters((self.da), name='Va')
        self.Wa_p = self.pc.add_parameters((self.da, self.di), name='Wa')
        self.Wha_p = self.pc.add_parameters((self.da, self.dh), name='Wha')

    def init(self, test=True, update=True):
        self.Va = self.Va_p.expr(update)
        self.Wa = self.Wa_p.expr(update)
        self.Wha = self.Wha_p.expr(update)

    def __call__(self, H, h, test=True):
        d = dy.tanh(dy.colwise_add(self.Wa * H, self.Wha * h))
        scores = dy.transpose(d) * self.Va
        weights = dy.softmax(scores)
        context = H * weights
        return context, weights


def get_attention(attention, di, dh, da, pc):
    if attention == 'empty':
        return EmptyAttention(pc)
    elif attention == 'mlp':
        return MLPAttention(di, dh, da, pc)
    else:
        print('Unknown attention type "%s", using mlp attention' % attention)
        return MLPAttention(di, dh, da, pc)
