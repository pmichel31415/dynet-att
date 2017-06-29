from __future__ import print_function, division

import numpy as np
import argparse
import yaml

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--dynet-seed", default=0, type=int)
parser.add_argument("--dynet-mem", default=512, type=int)
parser.add_argument("--dynet-gpu", help="Use dynet with GPU", action="store_true")
parser.add_argument("--dynet-autobatch", default=0, type=int, help="Use dynet autobatching")
parser.add_argument("--config_file", '-c',
                    default=None, type=str)
parser.add_argument("--env", '-e', help="Environment in the config file",
                    default='train', type=str)
parser.add_argument("--output_dir", '-od', help="Output directory", type=str, default='.')
parser.add_argument("--train_src", '-ts', help="Train data in the source language", type=str)
parser.add_argument("--train_dst", '-td', help="Train data in the target language", type=str)
parser.add_argument("--valid_src", '-vs', help="Validation data in the source language", type=str)
parser.add_argument("--valid_dst", '-vd', help="Validation data in the target language", type=str)
parser.add_argument("--test_src", '-tes', help="Test data in the source language", type=str)
parser.add_argument("--test_dst", '-ted', help="Test data in the target language", type=str)
parser.add_argument("--dic_src", '-dis',
                    help="File to save the source language dictionary to", type=str)
parser.add_argument("--dic_dst", '-did',
                    help="File to save the target language dictionary to", type=str)
parser.add_argument("--test_out", '-teo', help="File to save the translated test data", type=str)
parser.add_argument("--valid_out", '-vo',
                    help="File to save the translated validation data", type=str)
parser.add_argument("--lm_file", '-lmf', help="File to save the target language model", type=str)
parser.add_argument("--model", '-m', type=str,
                    help='Model file ([exp_name]_model if not specified)')
parser.add_argument("--trainer", '-tr', type=str,
                    help='Optimizer. Choose from "sgd,clr,momentum,adam,rmsprop"', default='sgd')
parser.add_argument('--num_epochs', '-ne', type=int, default=1,
                    help='Number of epochs (full pass over the training data) to train on')
parser.add_argument('--patience', '-p', type=int, default=0,
                    help='Patience before early stopping. No early stopping if <= 0')
parser.add_argument('--src_vocab_size', '-svs',
                    type=int, help='Maximum vocab size of the source language', default=40000)
parser.add_argument('--trg_vocab_size', '-tvs',
                    type=int, help='Maximum vocab size of the target language', default=20000)
parser.add_argument('--batch_size', '-bs',
                    type=int, help='minibatch size', default=20)
parser.add_argument('--dev_batch_size', '-dbs',
                    type=int, help='minibatch size for the validation set', default=10)
parser.add_argument('--num_layers', '-nl', type=int, default=1,
                    help='Number of layers in the encoder/decoder (For now only one is supported)')
parser.add_argument('--emb_dim', '-de',
                    type=int, help='Embedding dimension', default=256)
parser.add_argument('--att_dim', '-da',
                    type=int, help='Attention dimension', default=256)
parser.add_argument('--hidden_dim', '-dh',
                    type=int, help='Hidden dimension (for the recurrent networks)', default=256)
parser.add_argument('--label_smoothing', '-ls', type=float, default=0.0,
                    help='Label smoothing (interpolation coefficient with '
                    'the uniform distribution)')
parser.add_argument('--language_model', '-lm',
                    type=str, help='Language model to interpolate with', default=None)
parser.add_argument('--dropout_rate', '-dr',
                    type=float, help='Dropout rate', default=0.0)
parser.add_argument('--word_dropout_rate', '-wdr',
                    type=float, help='Word dropout rate', default=0.0)
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
parser.add_argument('--valid_bleu_every', '-vbe',
                    type=int, help='Compute BLEU on validation set every', default=500)
parser.add_argument('--max_len', '-ml', type=int,
                    help='Maximum length of generated sentences', default=60)
parser.add_argument('--beam_size', '-bm', type=int,
                    help='Beam size for beam search', default=1)
parser.add_argument('--bootstrap_number', '-bootn', type=int,
                    help='Number of samples for bootstrap', default=10)
parser.add_argument('--bootstrap_size', '-boots', type=int,
                    help='Size of subsets for bootstrap (in percentage)', default=50)
parser.add_argument('--min_freq', '-mf', type=int,
                    help='Minimum frequency under which words are unked', default=1)
parser.add_argument("--exp_name", '-en', type=str, default='experiment',
                    help='Name of the experiment (used so save the model)')
parser.add_argument("--bidir", '-bid',
                    help="Activates bidirectionnal encoding",
                    action="store_true")
parser.add_argument("--word_emb", '-we',
                    help="Activates direct word embedding for attention [currently deactivated]",
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
parser.add_argument("--retranslate",
                    help="Whether to retranslate the test data (true by default)",
                    action="store_false")


def parse_options():
    """Parse options from command line arguments and optionally config file

    Returns:
        Options
        argparse.Namespace
    """
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
    # Little trick : add dynet general options to sys.argv if they're not here
    # already. Linked to this issue : https://github.com/clab/dynet/issues/475
    if opt.dynet_gpu and '--dynet-gpus' not in sys.argv:
        sys.argv.append('--dynet-gpus')
        sys.argv.append('1')
    if '--dynet-autobatch' not in sys.argv:
        sys.argv.append('--dynet-autobatch')
        sys.argv.append(str(opt.__dict__['dynet_autobatch']))
    if '--dynet-mem' not in sys.argv:
        sys.argv.append('--dynet-mem')
        sys.argv.append(str(opt.__dict__['dynet_mem']))
    if '--dynet-seed' not in sys.argv:
        sys.argv.append('--dynet-seed')
        sys.argv.append(str(opt.__dict__['dynet_seed']))
        if opt.__dict__['dynet_seed'] > 0:
            np.random.seed(opt.__dict__['dynet_seed'])
    return opt


def print_config(opt, **kwargs):
    """Print the current configuration

    Prints command line arguments plus any kwargs

    Arguments:
        opt (argparse.Namespace): Command line arguments
        **kwargs: Any other key=value pair
    """
    print('======= CONFIG =======')
    for k, v in vars(opt).items():
        print(k, ':', v)
    for k, v in kwargs.items():
        print(k, ':', v)
    print('======================')


# Do this so sys.argv is changed upon import
options = parse_options()


def get_options():
    """Clean way to get options

    Returns:
        Options
        argparse.Namespace
    """
    return options
