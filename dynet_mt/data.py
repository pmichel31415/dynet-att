#!/usr/bin/env python3

import numpy as np
import dynn

from .util import Logger, default_filename
from .tokenizers import SubwordTokenizer


def append_to_file(sentences, filename):
    with open(filename, 'a+') as f:
        for s in sentences:
            print(s, file=f)


def dic_from_data(filename, tok, lang, min_count=1, max_size=-1, freeze=True):
    # Load from file
    file_content = dynn.io.loadtxt(filename)
    # Tokenize
    file_content = [tok.tokenize(line, lang)
                    for line in file_content]
    # Make dictionary
    dic = dynn.data.Dictionary.from_data(
        file_content,
        min_count=min_count,
        max_size=max_size
    )
    if freeze:
        dic.freeze()
    return dic


def dic_from_subword_vocab(filename):
    lines = dynn.io.loadtxt(filename)
    symbols = [line.split("\t")[0] for line in lines[3:]]
    return dynn.data.Dictionary(symbols=symbols)


def dictionaries_from_args(args, tok):
    if args.dic_src:
        # Load existing dictionary
        dic_src = dynn.data.Dictionary.load(args.dic_src)
    else:
        if isinstance(tok, SubwordTokenizer):
            # For subword models use the pre-computed dictionary
            vocab_file = f"{tok.prefix}.vocab"
            dic_subword = dic_from_subword_vocab(vocab_file)
            return dic_subword, dic_subword
        else:
            # Otherwise learn the dictionary from the data
            dic_src = dic_from_data(
                args.train_src,
                tok,
                args.src_lang,
                min_count=args.min_freq,
                max_size=args.src_vocab_size
            )
    if args.dic_tgt:
        dic_tgt = dynn.data.Dictionary.load(args.dic_tgt)
    else:
        dic_tgt = dic_from_data(
            args.train_tgt,
            tok,
            args.tgt_lang,
            min_count=args.min_freq,
            max_size=args.tgt_vocab_size
        )
    return dic_src, dic_tgt


def load_and_numberize(filename, dic, tok, lang, encoding="utf-8"):
    data = []
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            words = tok.tokenize(line.strip(), lang=lang)
            data.append(dic.numberize(words))
    return data


def append_eos(data, eos_token):
    for sample in data:
        sample.append(eos_token)
    return data


def prepare_training_data(args, tok, log=None):
    train_data_cache = args.train_data_cache
    if train_data_cache is None:
        train_data_cache = default_filename(args, "train.cache.bin")
    caching_func = dynn.data.caching.cached_to_file(train_data_cache)
    prepare_func = caching_func(_prepare_training_data)
    return prepare_func(
        args,
        tok,
        update_cache=args.clear_train_data_cache,
        log=log
    )


def _prepare_training_data(args, tok, log=None):
    # Log
    log = log or Logger()
    # Dictionaries
    log("Create dictionaries")
    dic_src, dic_tgt = dictionaries_from_args(args, tok)
    # Load training data
    log("Load training data")
    train_src = load_and_numberize(args.train_src, dic_src, tok, args.src_lang)
    train_tgt = load_and_numberize(args.train_tgt, dic_tgt, tok, args.tgt_lang)
    # Dev data
    log("Load validation data")
    valid_src = load_and_numberize(args.valid_src, dic_src, tok, args.src_lang)
    valid_tgt = load_and_numberize(args.valid_tgt, dic_tgt, tok, args.tgt_lang)

    # Append EOS to the target
    train_tgt = append_eos(train_tgt, dic_tgt.eos_idx)
    valid_tgt = append_eos(valid_tgt, dic_tgt.eos_idx)

    # Return as dictionary
    data = {"train_src": train_src, "train_tgt": train_tgt,
            "valid_src": valid_src, "valid_tgt": valid_tgt}
    # Return
    return data, dic_src, dic_tgt


def prepare_training_batches(args, dataset, dic_src, dic_tgt):
    train_batches = dynn.data.batching.SequencePairsBatches(
        dataset["train_src"],
        dataset["train_tgt"],
        dic_src,
        dic_tgt,
        max_samples=args.batch_size,
        max_tokens=args.max_tokens_per_batch,
    )
    valid_batches = dynn.data.batching.SequencePairsBatches(
        dataset["valid_src"],
        dataset["valid_tgt"],
        dic_src,
        dic_tgt,
        max_samples=args.valid_batch_size,
        max_tokens=args.max_tokens_per_valid_batch,
    )
    return train_batches, valid_batches


def _prepare_test_data(args, tok, log=None):
    # Log
    log = log or Logger()
    # Dictionaries
    dic_src, dic_tgt = dictionaries_from_args(args, tok)
    # Load test data
    test_src = load_and_numberize(args.test_src, dic_src, tok, args.src_lang)
    test_tgt = load_and_numberize(args.test_tgt, dic_tgt, tok, args.tgt_lang)

    # Append EOS to the target
    test_tgt = append_eos(test_tgt, dic_tgt.eos_idx)

    # Return as dictionary
    data = {"test_src": test_src, "test_tgt": test_tgt}
    # Return
    return data, dic_src, dic_tgt


def load_word_vectors(filename, dic):
    print('Reading word vectors from %s' % filename)
    non_zero = 0
    with open(filename, 'r') as f:
        # Read vector dimension in first line
        dim = int(f.readline().split()[1])
        vec = np.zeros((len(dic), dim))
        for l in f:
            word = l.split()[0].lower()
            if word in dic:
                non_zero += 1
                vector = np.asarray(l.split()[1:], dtype=float)
                vec[dic[word]] = vector
    print('Loaded %d pretrained word vectors (%.2f%%)' %
          (non_zero, 100 * non_zero / len(dic)))
    return vec
