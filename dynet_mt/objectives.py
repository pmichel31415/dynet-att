#!/usr/bin/env python3

import dynet as dy
import dynn


class Loss(object):
    """An object holding the loss function values"""

    def __init__(self, nll, ls_loss, loss):
        self.nll = nll
        self.ls_loss = ls_loss
        self.loss = loss


class Objective(dynn.layers.BaseLayer):

    def __call__(self, logits, tgt):
        raise NotImplementedError()


class NLLObjective(Objective):

    def __init__(self, label_smoothing=0.0):
        self.ls_eps = label_smoothing

    def __call__(self, logits, tgt):
        if len(logits) != len(tgt.sequences):
            raise ValueError("Target lengths and # logits mismatch")
        if self.ls_eps:
            logprobs = [dy.log_softmax(logit) for logit in logits]
            # NLL of the targets
            y_nlls = [-dy.pick_batch(logprob, y)
                      for logprob, y in zip(logprobs, tgt.sequences)]
            # NLL of the uniform distribution
            uniform_nlls = [-dy.mean_elems(logprob) for logprob in logprobs]
            # Average with smoothing
            nlls = [(1 - self.ls_eps) * y_nll + self.ls_eps * uniform_nll
                    for y_nll, uniform_nll in zip(y_nlls, uniform_nlls)]
        else:
            nlls = [dy.pickneglogsoftmax_batch(logit, y)
                    for logit, y in zip(logits, tgt.sequences)]
        # Masking
        masked_nll = dynn.operations.stack(nlls, d=-1) * tgt.get_mask()
        # Reduce
        nll = dy.sum_batches(masked_nll) / sum(tgt.lengths)
        return nll


def objective_from_args(args):
    if args.objective == "nll":
        return NLLObjective(args.label_smoothing)
    else:
        raise ValueError(f"Unknown objective \"{args.objective}\"")
