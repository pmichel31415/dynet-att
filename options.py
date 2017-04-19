from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--dynet-seed", default=0, type=int)
parser.add_argument("--dynet-mem", default=512, type=int)
parser.add_argument("--dynet-gpus", default=0, type=int)
parser.add_argument("--config_file", '-c',
                    default=None, type=str)
parser.add_argument("--env", '-e',
                    default='train', type=str)
parser.add_argument("--train_src", '-ts',
                    default='en-de/train.en-de.de', type=str)
parser.add_argument("--train_dst", '-td',
                    default='en-de/train.en-de.en', type=str)
parser.add_argument("--valid_src", '-vs',
                    default='en-de/valid.en-de.de', type=str)
parser.add_argument("--valid_dst", '-vd',
                    default='en-de/valid.en-de.en', type=str)
parser.add_argument("--test_src", '-tes',
                    default='en-de/test.en-de.de', type=str)
parser.add_argument("--test_dst", '-ted',
                    default='en-de/test.en-de.en', type=str)
parser.add_argument("--dic_src", '-dis',
                    default=None, type=str)
parser.add_argument("--dic_dst", '-did',
                    default=None, type=str)
parser.add_argument("--test_out", '-teo',
                    default='results/out.en-de.en', type=str)
parser.add_argument("--model", '-m', type=str, help='Model to load from')
parser.add_argument("--trainer", '-tr', type=str, help='Optimizer', default='sgd')
parser.add_argument('--num_epochs', '-ne',
                    type=int, help='Number of epochs', default=1)
parser.add_argument('--src_vocab_size', '-svs',
                    type=int, help='Maximum vocab size of the source language', default=40000)
parser.add_argument('--trg_vocab_size', '-tvs',
                    type=int, help='Maximum vocab size of the target language', default=20000)
parser.add_argument('--batch_size', '-bs',
                    type=int, help='minibatch size', default=20)
parser.add_argument('--dev_batch_size', '-dbs',
                    type=int, help='minibatch size for the validation set', default=10)
parser.add_argument('--emb_dim', '-de',
                    type=int, help='embedding size', default=256)
parser.add_argument('--att_dim', '-da',
                    type=int, help='attention size', default=256)
parser.add_argument('--hidden_dim', '-dh',
                    type=int, help='hidden size', default=256)
parser.add_argument('--dropout_rate', '-dr',
                    type=float, help='dropout rate', default=0.0)
parser.add_argument('--gradient_clip', '-gc', type=float, default=1.0,
                    help='Gradient clipping. Negative value means no clipping')
parser.add_argument('--learning_rate', '-lr',
                    type=float, help='learning rate', default=1.0)
parser.add_argument('--learning_rate_decay', '-lrd',
                    type=float, help='learning rate decay', default=0.0)
parser.add_argument('--check_train_error_every', '-ct',
                    type=int, help='Check train error every', default=100)
parser.add_argument('--check_valid_error_every', '-cv',
                    type=int, help='Check valid error every', default=1000)
parser.add_argument('--test_every', '-te',
                    type=int, help='Run on test set every', default=500)
parser.add_argument('--max_len', '-ml', type=int,
                    help='Maximum length of generated sentences', default=60)
parser.add_argument('--beam_size', '-bm', type=int,
                    help='Beam size for beam search', default=1)
parser.add_argument("--exp_name", '-en', type=str, default='experiment',
                    help='Name of the experiment (used so save the model)')
parser.add_argument("--bidir", '-bid',
                    help="Activates bidirectionnal encoding",
                    action="store_true")
parser.add_argument("--word_emb", '-we',
                    help="Activates direct word embedding for attention",
                    action="store_true")
parser.add_argument("--verbose", '-v',
                    help="increase output verbosity",
                    action="store_true")
parser.add_argument("--train",
                    help="Print debugging info",
                    action="store_true")
parser.add_argument("--test",
                    help="Print debugging info",
                    action="store_true")


def get_options():
    opt = parser.parse_args()
    if opt.config_file:
        with open(opt.config_file, 'r') as f:
            data = yaml.load(f)
            delattr(opt, 'config_file')
            arg_dict = opt.__dict__
            for key, value in data.items():
                if isinstance(value, dict):
                    if key == opt.env:
                        for k, v in value.items():
                            arg_dict[k] = v
                    else:
                        continue
                else:
                    arg_dict[key] = value

    return opt


def print_config(opt, **kwargs):
    print('======= CONFIG =======')
    for k, v in vars(opt).items():
        print(k, ':', v)
    for k, v in kwargs.items():
        print(k, ':', v)
    print('======================')