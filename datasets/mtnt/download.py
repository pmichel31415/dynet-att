#!/usr/bin/env python3
import os
import re
import tarfile
import argparse

from dynn.data.data_util import download_if_not_there

HERE = os.path.dirname(os.path.realpath(__file__))

mtnt_url = "https://github.com/pmichel31415/mtnt/releases/download/v1.0/"

pretrain_files = {
    "ja-en": "clean-data-en-ja.tar.gz",
    "en-ja": "clean-data-en-ja.tar.gz",
    "fr-en": "clean-data-en-fr.tar.gz",
    "en-fr": "clean-data-en-fr.tar.gz",
}

mtnt_file = "MTNT.1.0.tar.gz"


test_filename = re.compile(
    r"IWSLT[0-9]+\.TED\."
    r"([a-z0-9]+\.[a-z\-]+\.[a-z]+)\.xml"
)
train_filename = re.compile(r"train\.tags\.([a-z\-]+\.[a-z]+)")


def get_args():
    parser = argparse.ArgumentParser("Download the IWSLT dataset")
    parser.add_argument("--langpair", type=str, required=True,
                        choices=["ja-en", "en-ja", "fr-en", "en-fr"])
    parser.add_argument("--pretrain-data", action="store_true",
                        help="Download the pretraining data (\"clean data\")")
    parser.add_argument("--all-data", action="store_true",
                        help="Download all the data (pretraining and MTNT)")
    parser.add_argument("--re-download", action="store_true")
    return parser.parse_args()


def download_pretrain(langpair, re_download):
    download_if_not_there(
        pretrain_files[langpair],
        mtnt_url,
        HERE,
        force=re_download
    )

    abs_filename = os.path.join(HERE, pretrain_files[langpair])
    # Create target dir
    directory = "pretrain"
    root_path = os.path.join(HERE, directory)
    if not os.path.isdir(root_path):
        os.mkdir(root_path)
    # Extract
    with tarfile.open(abs_filename) as tar:
        tar.extractall(root_path)


def download_mtnt(langpair, re_download):
    download_if_not_there(
        mtnt_file,
        mtnt_url,
        HERE,
        force=re_download
    )

    src_lang, tgt_lang = langpair.split("-")
    abs_filename = os.path.join(HERE, mtnt_file)
    # Create target dir
    root_path = HERE
    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    def members(tf):
        train_prefix = "MTNT/train/"
        valid_prefix = "MTNT/valid/"
        test_prefix = "MTNT/test/"
        mono_prefix = "MTNT/mono/"
        members = []
        for member in tf.getmembers():
            if member.path.startswith(f"{train_prefix}train.{langpair}"):
                member.path = member.path[len(train_prefix):]
                members.append(member)
            elif member.path.startswith(f"{valid_prefix}valid.{langpair}"):
                member.path = member.path[len(valid_prefix):]
                members.append(member)
            elif member.path.startswith(f"{test_prefix}test.{langpair}"):
                member.path = member.path[len(test_prefix):]
                members.append(member)
            elif member.path.startswith(f"{mono_prefix}mono.{src_lang}"):
                member.path = member.path[len(mono_prefix):]
                members.append(member)
            elif member.path.startswith(f"{mono_prefix}mono.{tgt_lang}"):
                member.path = member.path[len(mono_prefix):]
                members.append(member)
        return members
    # Extract
    with tarfile.open(abs_filename) as tar:
        tar.extractall(root_path, members=members(tar))
    # Split tsv files
    for filename in os.listdir(root_path):
        if filename.endswith(".tsv"):
            full_prefix = os.path.join(root_path, filename[:-4])
            with open(f"{full_prefix}.tsv", "r", encoding="utf-8") as tsv_file:
                src_file = open(f"{full_prefix}.{src_lang}",
                                "w", encoding="utf-8")
                tgt_file = open(f"{full_prefix}.{tgt_lang}",
                                "w", encoding="utf-8")
                for tsv_line in tsv_file:
                    _, src_line, tgt_line = tsv_line.split("\t")
                    print(src_line, file=src_file)
                    print(tgt_line, file=tgt_file)
                src_file.close()
                tgt_file.close()
    # Remove tsvs
    for filename in os.listdir(root_path):
        if filename.endswith(".tsv"):
            os.remove(os.path.join(root_path, filename))


def main():
    args = get_args()
    if args.pretrain_data:
        download_pretrain(args.langpair, args.re_download)
    elif args.all_data:
        download_mtnt(args.langpair, args.re_download)
        download_pretrain(args.langpair, args.re_download)
    else:
        download_mtnt(args.langpair, args.re_download)


if __name__ == "__main__":
    main()
