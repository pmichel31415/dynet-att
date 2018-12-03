#!/usr/bin/env python3
from dynn import io

from .bilstm import AttBiLSTM
from .transformer import Transformer
from ..util import default_filename

from .architectures import architecture_from_args


model_types = {"bilstm": AttBiLSTM, "transformer": Transformer}
supported_model_types = list(model_types.keys())


def assert_model_type_supported(model_type):
    if model_type not in model_types:
        raise ValueError(
            f"Unknown model type {model_type}. "
            f"Supported model types are {', '.join(supported_model_types)}"
        )


def add_model_type_args(model_type, parser):
    """Add model specific arguments to the parser"""
    assert_model_type_supported(model_type)
    model_types[model_type].add_args(parser)


def model_from_args(args, dic_src, dic_tgt):
    """Return a model from command line arguments"""
    # Check if we should produce an existing architecture
    if getattr(args, "architecture", None) is not None:
        return architecture_from_args(args, dic_src, dic_tgt)
    # Check model type
    model_type = args.model_type
    assert_model_type_supported(model_type)
    # Build model
    model = model_types[model_type].from_args(args, dic_src, dic_tgt)
    # Load model maybe
    if args.pretrained_model:
        if args.model_load_file is None:
            setattr(args, "model_load_file",
                    default_filename(args, "model.npz"))
        io.populate(model.pc, args.model_load_file)
    return model
