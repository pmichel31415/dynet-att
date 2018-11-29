#!/usr/bin/env python3

import dynet as dy
import dynn


class Optimizer(object):

    def __init__(self, trainer, lr_schedule):
        self.trainer = trainer
        self.lr_schedule = lr_schedule
        self.lr_schedule_value = next(self.lr_schedule)
        self.step = 0
        self.lr_scale = 1.0

    def update(self):
        self.trainer.learning_rate = self._learning_rate(step=True)
        self.trainer.update()

    def _learning_rate(self, step=True):
        if step:
            # Ignore the 1st step because we already called next
            if self.step > 0:
                self.lr_schedule_value = next(self.lr_schedule)
            self.step += 1
        return self.lr_schedule_value * self.lr_scale

    @property
    def learning_rate(self):
        # Read only, don't step
        return self._learning_rate(step=False)

    def decay_lr(self, factor):
        self.lr_scale = factor * self.lr_scale


dynet_trainers = {
    "sgd": dy.SimpleSGDTrainer,
    "adam": dy.AdamTrainer,
    "amsgrad": dy.AmsgradTrainer,
}


def lr_schedule_from_args(args):
    schedule_name = getattr(args, "lr_schedule", "constant")
    if schedule_name == "constant":
        def constant_schedule():
            while True:
                yield args.learning_rate
        return constant_schedule()
    elif schedule_name == "inverse_sqrt":
        warmup = getattr(args, "lr_warmup", 0.1)  # default to no warmup
        return dynn.training.inverse_sqrt_schedule(warmup, args.learning_rate)


def trainer_from_args(args, pc):
    momentum = getattr(args, "momentum", 0)
    if args.trainer == "sgd":
        if momentum:
            trainer = dy.MomentumSGDTrainer(
                pc,
                learning_rate=args.learning_rate,
                mom=momentum
            )
        else:
            trainer = dy.SimpleSGDTrainer(pc, learning_rate=args.learning_rate)
    elif args.trainer == "adam":
        trainer = dy.AdamTrainer(
            pc,
            alpha=args.learning_rate,
            beta1=args.momentum
        )
    elif args.trainer == "amsgrad":
        trainer = dy.AdamTrainer(
            pc,
            alpha=args.learning_rate,
            beta1=args.momentum
        )
    return trainer


def optimizer_from_args(args, pc):
    trainer = trainer_from_args(args, pc)
    lr_schedule = lr_schedule_from_args(args)
    return Optimizer(trainer, lr_schedule)
