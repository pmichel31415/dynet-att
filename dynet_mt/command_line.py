#!/usr/bin/env python3
import argparse
import yaml

import dynn
from .models import supported_model_types
from .models.architectures import supported_architectures
from .tokenizers import supported_tokenizer_types

from .util import Logger


def get_base_parser(with_tasks=True):
    parser = argparse.ArgumentParser()
    # Add dynet args
    dynn.command_line.add_dynet_args(parser)
    parser.add_argument("--config-file",
                        default=None, type=str)
    parser.add_argument("--env", help="Environment in the config file",
                        default="train", type=str)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("--exp-name", type=str, default="test",
                        help="Name of the experiment (used so save the model)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--temp-dir", type=str, default="temp",
                        help="Temp directory")
    parser.add_argument("--src-lang", type=str, help="Source language")
    parser.add_argument("--tgt-lang", type=str, help="Target language")
    if with_tasks:
        tasks_parsers = parser.add_subparsers(title="Tasks", dest="task")
        return parser, tasks_parsers
    else:
        return parser


def add_preprocessing_args(parser):
    prep_group = parser.add_argument_group("Preprocessing")
    prep_group.add_argument("--lowercase", action="store_true")
    prep_group.add_argument("--src-vocab-size", default=40000, type=int,
                            help="Maximum vocab size of the source language")
    prep_group.add_argument("--tgt-vocab-size", default=40000, type=int,
                            help="Maximum vocab size of the target language")
    prep_group.add_argument("--dic-src", type=str,
                            help="File containing the dictionary in the source"
                            " language")
    prep_group.add_argument("--dic-tgt", type=str,
                            help="File containing the dictionary in the target"
                            " language")
    prep_group.add_argument("--min-freq", type=int, default=1,
                            help="Minimum frequency under which words are "
                            "unked")
    prep_group.add_argument("--shared-vocab", action="store_true",
                            help="Share vocabulary between source and target")
    prep_group.add_argument("--tokenizer-file", type=str,
                            help="File where the tokenizer will be saved "
                            "(if the file exist, load the tokenizer from "
                            "there directly)")
    prep_group.add_argument("--tokenizer-type", type=str,
                            help="Type of tokenizer",
                            choices=supported_tokenizer_types)


def add_translation_args(parser):
    trans_group = parser.add_argument_group("Translation")
    trans_group.add_argument("--lex-s2t", type=str, help="File containing a "
                             "lexicon from source to target")
    trans_group.add_argument("--lex-t2s", type=str, help="File containing a "
                             "lexicon from target to source")
    trans_group.add_argument("--max-len", type=int, default=9999,
                             help="Maximum length of generated sentences")
    trans_group.add_argument("--beam-size", type=int, default=1,
                             help="Beam size for beam search")
    trans_group.add_argument("--lenpen", type=int, default=0.0,
                             help="Length penalty for beam search")
    trans_group.add_argument("--replace-unk", action="store_true")
    trans_group.add_argument("--synonyms-file", type=str,
                             help="File with synonyms for each word")


def add_optimization_args(parser):
    optim_group = parser.add_argument_group("Optimization")
    # Hyper-parameters
    optim_group.add_argument("--objective", type=str, default="nll")
    optim_group.add_argument("--label-smoothing", type=float, default=0.0)
    optim_group.add_argument("--trainer", type=str, default="sgd")
    optim_group.add_argument("--n-epochs", type=int, default=1)
    optim_group.add_argument("--patience", type=int, default=0)
    optim_group.add_argument("--batch-size", type=int, default=32)
    optim_group.add_argument("--max-tokens-per-batch", type=int, default=9999)
    optim_group.add_argument("--valid-batch-size", type=int, default=10)
    optim_group.add_argument("--max-tokens-per-valid-batch", type=int,
                             default=9999)
    optim_group.add_argument("--gradient-clip", type=float, default=1.0,
                             help="Gradient clipping. Negative value "
                             "means no clipping")
    optim_group.add_argument("--learning-rate", type=float, default=1.0)
    optim_group.add_argument("--learning-rate-decay", type=float, default=1.0)
    optim_group.add_argument("--learning-rate-schedule", type=str,
                             choices=["constant", "inverse_sqrt"],
                             default="constant",)
    optim_group.add_argument("--learning-rate-warmup", type=int, default=1.0)
    optim_group.add_argument("--momentum", type=float, default=0.0)
    optim_group.add_argument("--report-every", type=int, default=100,
                             help="Check train error every")
    optim_group.add_argument("--valid-every", type=int, default=1000,
                             help="Check valid error every")
    optim_group.add_argument("--valid-bleu-every", type=int, default=0,
                             help="Compute BLEU on validation set every")


def add_model_args(parser):
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model-type", type=str, default="bilstm",
                             help="Model type",
                             choices=supported_model_types)
    model_group.add_argument("--architecture", type=str, default=None,
                             help="Use a predefined architecture",
                             choices=supported_architectures)
    model_group.add_argument("--model-save-file", type=str,
                             help="File where the model will be saved. if "
                             "this is not specified the model will be saved "
                             "to [output_dir]/[exp_name].model.npz")
    model_group.add_argument("--pretrained-model",
                             help="Whether to use a pretrained model. If "
                             "\"--model-load-file\" is not specified this will"
                             " attempt to load from [output_dir]/[exp_name]."
                             "model.npz",
                             action="store_true")
    model_group.add_argument("--model-load-file", type=str,
                             help="File from whence the model should be "
                             "loaded. This will only be used if "
                             "\"--pretrained-model\" is used.")
    model_group.add_argument("--pretrained-src-wembs", type=str,
                             help="Pretrained source word embeddings file")
    model_group.add_argument("--pretrained-tgt-wembs", type=str,
                             help="Pretrained target word embeddings")


def add_evaluation_args(parser):
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--bootstrap-num-samples", type=int, default=100,
                            help="Number of samples for bootstrap resampling")
    eval_group.add_argument("--bootstrap-sample-size", type=float, default=50,
                            help="Size of each sample (in percentage of the "
                            "total size)")


def parse_args_and_yaml(parser, known_args_only=True):
    """Parse options from command line arguments and optionally config file"""
    if known_args_only:
        args = parser.parse_args()
    else:
        args, _ = parser.parse_known_args()
    # Parse config file
    if args.config_file:
        with open(args.config_file, "r") as f:
            # Read general and environment specific arguments from yaml
            data = yaml.load(f)
            gen_args = {}
            env_args = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    if key == args.env:
                        for k, v in value.items():
                            env_args[k] = v
                    else:
                        continue
                else:
                    gen_args[key] = value

            # Add those arguments to the args namespace
            add_to_args(args, gen_args, known_args_only=known_args_only)
            # env args are added last so that they override everything
            add_to_args(args, env_args, known_args_only=known_args_only)
    return args


def add_to_args(args, dic, known_args_only=True):
    arg_dict = args.__dict__
    for key, value in dic.items():
        if known_args_only and key not in arg_dict:
            raise ValueError(
                f"Unknown argument {key} in config file "
                f"{args.config_file}"
            )
        else:
            arg_dict[key] = value


def print_config(args, log=None, **kwargs):
    """Print the current configuration as yaml

    Prints command line arguments plus any kwargs

    Arguments:
        args (argparse.Namespace): Command line arguments
        **kwargs: Any other key=value pair
    """
    log = log or Logger()
    log("####### CONFIG #######")
    for k, v in vars(args).items():
        log(f"{k}: {v}")
    for k, v in kwargs.items():
        log(f"{k}: {v}")
    log("######################")
