from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

class Beam(object):
    def __init__(self, state, context, words, logprob):
        self.state = state
        self.words = words
        self.context = context
        self.logprob = logprob