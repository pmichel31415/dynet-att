#!/usr/bin/env python3
import os
import re

import argparse
from dynn.data import iwslt
from dynn import io

HERE = os.path.dirname(os.path.realpath(__file__))

test_filename = re.compile(
    r"IWSLT[0-9]+\.TED\."
    r"([a-z0-9]+\.[a-z\-]+\.[a-z]+)\.xml"
)
train_filename = re.compile(r"train\.tags\.([a-z\-]+\.[a-z]+)")


def get_args():
    parser = argparse.ArgumentParser("Download the IWSLT dataset")
    parser.add_argument("--langpair", type=str, required=True)
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--re-download", action="store_true")
    return parser.parse_args()


def preprocess_test(xml_file):
    xml = io.loadtxt(xml_file)
    txt = []
    for line in xml:
        seg = iwslt.eval_segment.match(line)
        if seg:
            txt.append(seg.group(1).strip())
    return txt


def preprocess_train(xml_file):
    xml = io.loadtxt(xml_file)
    txt = []
    for line in xml:
        meta_line = iwslt.is_meta.match(line)
        if meta_line:
            continue
        txt.append(line)
    return txt


def main():
    args = get_args()
    iwslt.download_iwslt(
        HERE,
        year=args.year,
        langpair=args.langpair,
        force=args.re_download
    )
    folder = os.path.join(
        HERE,
        iwslt.local_dir(args.year, args.langpair)
    )
    for filename in os.listdir(folder):
        is_test = test_filename.match(filename)
        is_train = train_filename.match(filename)
        if is_test:
            txt = preprocess_test(os.path.join(folder, filename))
            io.savetxt(os.path.join(folder, is_test.group(1)), txt)
        elif is_train:
            txt = preprocess_train(os.path.join(folder, filename))
            io.savetxt(os.path.join(folder, f"train.{is_train.group(1)}"), txt)


if __name__ == "__main__":
    main()
