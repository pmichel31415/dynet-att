#!/usr/bin/env python3
import os.path
from .space_tokenizer import SpaceTokenizer
from .moses_tokenizer import MosesTokenizer
from .subword_tokenizer import SubwordTokenizer

from ..util import default_filename

tokenizer_types = {
    "space": SpaceTokenizer,
    "moses": MosesTokenizer,
    "subword": SubwordTokenizer,
}
supported_tokenizer_types = list(tokenizer_types.keys())


def assert_tokenizer_type_supported(tokenizer_type):
    if tokenizer_type not in tokenizer_types:
        raise ValueError(
            f"Unknown tokenizer type {tokenizer_type}. "
            "Supported tokenizer types are "
            f"{', '.join(supported_tokenizer_types)}"
        )


def add_tokenizer_args(tokenizer_type, parser):
    """Add tokenizer specific arguments to the parser"""
    assert_tokenizer_type_supported(tokenizer_type)
    tokenizer_types[tokenizer_type].add_args(parser)


def tokenizer_from_args(args):
    """Return a tokenizer from command line arguments"""
    tokenizer_type = getattr(args, "tokenizer_type", "space")
    assert_tokenizer_type_supported(tokenizer_type)
    if args.tokenizer_file is None:
        tokenizer_file = default_filename(args, "tokenizer")
    # Load the tokenizer if it exists
    if os.path.isfile(tokenizer_file):
        return tokenizer_types[tokenizer_type].load(tokenizer_file)
    else:
        tokenizer = tokenizer_types[tokenizer_type].from_args(args)
        tokenizer.save(tokenizer_file)
        return tokenizer
