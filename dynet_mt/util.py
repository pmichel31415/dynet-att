#!/usr/bin/env python3

import time
import sys
import os.path


class Logger(object):

    def __init__(self, verbose=True, out_file=None):
        if out_file is None:
            self.file = sys.stdout
        else:
            self.file = open(out_file, "w")
        self.verbose = verbose

    def __call__(self, string):
        if self.verbose:
            print(string, file=self.file)
            sys.stdout.flush()


class Timer(object):

    def __init__(self, verbose=False):
        self.start = time.time()

    def restart(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def tick(self):
        elapsed = self.elapsed()
        self.restart()
        return elapsed


def default_filename(args, suffix):
    prefix = os.path.abspath(os.path.join(args.output_dir, args.exp_name))
    return f"{prefix}.{suffix}"


def exp_temp_filename(opt, name):
    return opt.temp_dir + '/' + opt.exp_name + '_' + name
