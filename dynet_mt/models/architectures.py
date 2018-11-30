#!/usr/bin/env python3

from .bilstm import AttBiLSTM
from .transformer import Transformer

architectures = {
    # BiLSTM architectures
    "test_bilstm": {
        # This one is for testing
        "class": AttBiLSTM,
        "n_layers": 1,
        "embed_dim": 2,
        "hidden_dim": 3,
        "dropout": 0.1,
        "tie_decoder_embeds": True,
        "tie_all_embeds": False,
    },
    "small_bilstm": {
        "class": AttBiLSTM,
        "n_layers": 1,
        "embed_dim": 256,
        "hidden_dim": 256,
        "dropout": 0.2,
        "tie_decoder_embeds": True,
        "tie_all_embeds": False,
    },
    "medium_bilstm": {
        "class": AttBiLSTM,
        "n_layers": 2,
        "embed_dim": 256,
        "hidden_dim": 512,
        "dropout": 0.2,
        "tie_decoder_embeds": True,
        "tie_all_embeds": False,
    },
    # Transformers
    "test_transformer": {
        # This one is for testing
        "class": Transformer,
        "n_layers": 2,
        "embed_dim": 4,
        "hidden_dim": 4,
        "n_heads": 2,
        "dropout": 0.1,
        "tie_decoder_embeds": True,
        "tie_all_embeds": False,
    },
}
supported_architectures = list(architectures.keys())


def assert_architecture_supported(architecture_name):
    if architecture_name not in architectures:
        raise ValueError(
            f"Unknown architecture {architecture_name}. "
            f"Supported architecture types are "
            f"{', '.join(supported_architectures)}"
        )


def architecture_from_args(args, dic_src, dic_tgt):
    """Add model specific arguments to the parser"""
    assert_architecture_supported(args.architecture)
    for k, v in architectures[args.architecture].items():
        if k != "class":
            setattr(args, k, v)
    architecture_class = architectures[args.architecture]["class"]
    return architecture_class.from_args(args, dic_src, dic_tgt)
