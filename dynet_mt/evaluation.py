#!/usr/bin/env python3

import argparse
import numpy as np

from sacrebleu import corpus_bleu

from dynn import io
from . import util


def bleu_to_str(bleu):
    """Convert sacrebleu BLEU object to a moses style string"""
    precisions = "/".join(f"{p:.1f}" for p in bleu.precisions)
    string = (
        f"BLEU = {bleu.score:.2f} {precisions} (BP={bleu.bp: .3f} "
        f"ratio={bleu.sys_len/bleu.ref_len:.3f} hyp_len={bleu.sys_len:d} "
        f"ref_len={bleu.ref_len:d})"
    )
    return string


def bleu_score(ref_file, hyp_file, tokenize="intl"):
    """Computes corpus level BLEU score with sacreBLEU

    Arguments:
        ref_file (str): Path to the reference file
        hyp_file (str): Path to the hypothesis file

    Returns:
        tuple: Tuple (BLEU, details) containing the bleu score
            and the detailed output of the perl script

    Raises:
        ValueError: Raises error if the perl script fails for some reason
    """
    ref = io.loadtxt(ref_file)
    hyp = io.loadtxt(hyp_file)
    bleu = corpus_bleu(hyp, [ref], tokenize=tokenize)
    return bleu.score, bleu_to_str(bleu)


def get_args():
    parser = argparse.ArgumentParser(
        description="program to compare mt results",
    )
    parser.add_argument("ref", type=str,
                        help="a path to a correct reference file")
    parser.add_argument("out", type=str, help="a path to a system output")
    parser.add_argument("otherout", nargs="?", type=str, default=None,
                        help="a path to another system output. add only if "
                        "you want to compare outputs from two systems.")
    parser.add_argument("--num_samples", "-M", type=int, default=100,
                        help="Number of samples for bootstrap resampling")
    parser.add_argument("--sample_size", type=float, default=50,
                        help="Size of each sample (in %% of the total size)")
    parser.add_argument("--bleufile", type=str, default="bleus.txt",
                        help="Where to store the bleu scores")
    parser.add_argument("--verbose", "-v",
                        help="increase output verbosity",
                        action="store_true")


def print_stats(bleus):
    print("Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f" %
          (bleus.mean(), bleus.std(), bleus.min(), bleus.max()))
    print("95%% confidence interval: %.3f - %.3f" %
          (bleus[int(0.025 * len(bleus))], bleus[int(0.975 * len(bleus))]))


def paired_stats(bleus):
    N, _ = bleus.shape
    win1 = (bleus[:, 0] > bleus[:, 1]).sum() / N
    win2 = (bleus[:, 0] < bleus[:, 1]).sum() / N
    ties = (bleus[:, 0] == bleus[:, 1]).sum() / N
    return win1, win2, ties


def print_paired_stats(bleus, log=None):
    log = log or util.Logger()
    win1, win2, ties = paired_stats(bleus)
    log(f"System 1 > system 2: {win1*100:.2f}")
    log(f"System 1 < system 2: {win2*100:.2f}")
    log(f"Ties: {ties*100:.2f}")


def to_np_array(corpus):
    if isinstance(corpus, str):
        corpus = io.loadtxt(corpus)
    return np.asarray(corpus)


def bootstrap_resampling(ref, out, num_samples, sample_percent, log=None):
    log = log or util.Logger()
    ref = to_np_array(ref)
    out = to_np_array(out)
    n = len(ref)
    if n != len(out):
        raise ValueError("Mismatched reference and output file size")
    k = int(sample_percent * n / 100)
    bleus = []
    for i in range(num_samples):
        subset = np.random.choice(n, k)
        bleu, _ = bleu_score(out[subset], ref[subset])
        bleus.append(bleu)
        if (i + 1) % (num_samples // 10) == 0:
            log(f"{(i + 1) // (num_samples // 10) * 10}% done")
    bleus = np.sort(np.asarray(bleus))
    return bleus


def paired_bootstrap_resampling(
    ref,
    out1,
    out2,
    num_samples,
    sample_percent,
    tokenize="intl",
    log=None
):
    log = log or util.Logger()
    ref = to_np_array(ref)
    out1 = to_np_array(out1)
    out2 = to_np_array(out2)
    n = len(ref)
    if n != len(out1):
        raise ValueError("Mismatched reference and output file size")
    if n != len(out2):
        raise ValueError("Mismatched reference and other output file size")
    k = int(sample_percent * n / 100)
    bleus = []
    for i in range(num_samples):
        subset = np.random.choice(n, k)
        bleu1 = corpus_bleu(out1[subset], [ref[subset]], tokenize=tokenize)
        bleu2 = corpus_bleu(out2[subset], [ref[subset]], tokenize=tokenize)
        bleus.append([bleu1, bleu2])
        if (i + 1) % (num_samples // 10) == 0:
            log(f"{(i + 1) // (num_samples // 10) * 10}% done")
    bleus = np.asarray(bleus)
    return bleus


def main():
    args = get_args()
    log = util.Logger(verbose=args.verbose)
    ref = to_np_array(args.ref)
    out = to_np_array(args.out)
    if args.otherout is None:
        # Normal bootstrap resampling
        bleus = bootstrap_resampling(ref, out, args.num_samples,
                                     args.sample_size, log=log)
        total = bleu_score(args.ref, args.out)
        log("Total BLEU: %.3f" % total)
        print_stats(bleus)
        np.io.savetxt(args.bleufile, bleus)
    else:
        otherout = np.asarray(io.loadtxt(args.otherout))
        bleus = bootstrap_resampling(ref, out, otherout,
                                     args.num_samples,
                                     args.sample_size,
                                     verbose=args.verbose)
        print_paired_stats(bleus)
        np.io.savetxt(args.bleufile, bleus)


if __name__ == "__main__":
    main()
