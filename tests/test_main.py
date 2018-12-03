#!/usr/bin/env python3

import unittest
from unittest import TestCase

import os.path
import tempfile
import shutil
import sys
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(HERE, ".."))
from dynet_mt import main  # noqa


class TestMain(TestCase):

    def setUp(self):
        self.actual_sysargs = sys.argv[:]
        sys.argv = [sys.argv[0]]
        self.path = tempfile.mkdtemp()
        self.data_folder = os.path.join(HERE, "data")
        self.config_folder = os.path.join(HERE, "config")

    def tearDown(self):
        shutil.rmtree(self.path)
        sys.argv = self.actual_sysargs[:]

    def test_train_bilstm(self):
        train_src = os.path.join(self.data_folder, "train.ja")
        train_tgt = os.path.join(self.data_folder, "train.en")
        valid_src = os.path.join(self.data_folder, "valid.ja")
        valid_tgt = os.path.join(self.data_folder, "valid.en")
        sys.argv.extend([
            # General args
            "--exp-name", "test",
            "--output-dir", self.path,
            "--verbose",
            # Task
            "train",
            # Tokenizer args
            "--tokenizer-type", "subword",
            "--subword-algo", "unigram",
            "--subword-train-files", train_src, train_tgt,
            "--subword-voc-size", "1000",
            # Data args
            "--train-src", train_src,
            "--train-tgt", train_tgt,
            "--valid-src", valid_src,
            "--valid-tgt", valid_tgt,
            # Model args
            "--architecture", "test_bilstm",
            # Optimization args
            "--batch-size", "2",
            "--valid-batch-size", "2",
        ])
        main.main()

    def test_eval_ppl_bilstm(self):
        train_src = os.path.join(self.data_folder, "train.ja")
        train_tgt = os.path.join(self.data_folder, "train.en")
        valid_src = os.path.join(self.data_folder, "valid.ja")
        valid_tgt = os.path.join(self.data_folder, "valid.en")
        sys.argv.extend([
            # General args
            "--exp-name", "test",
            "--output-dir", self.path,
            "--verbose",
            # Task
            "train",
            # Tokenizer args
            "--tokenizer-type", "subword",
            "--subword-algo", "unigram",
            "--subword-train-files", train_src, train_tgt,
            "--subword-voc-size", "1000",
            # Data args
            "--train-src", train_src,
            "--train-tgt", train_tgt,
            "--valid-src", valid_src,
            "--valid-tgt", valid_tgt,
            # Model args
            "--architecture", "test_bilstm",
            # Optimization args
            "--batch-size", "2",
            "--valid-batch-size", "2",
        ])
        main.main()
        sys.argv = [sys.argv[0]]
        sys.argv.extend([
            # General args
            "--exp-name", "test",
            "--output-dir", self.path,
            "--verbose",
            # Task
            "eval_ppl",
            # Tokenizer args
            "--tokenizer-type", "subword",
            # Data args
            "--eval-src", valid_src,
            "--eval-tgt", valid_tgt,
            "--eval-batch-size", "2",
            # Model args
            "--architecture", "test_bilstm",
        ])

    def test_translate_bilstm(self):
        train_src = os.path.join(self.data_folder, "train.ja")
        train_tgt = os.path.join(self.data_folder, "train.en")
        valid_src = os.path.join(self.data_folder, "valid.ja")
        valid_tgt = os.path.join(self.data_folder, "valid.en")
        sys.argv.extend([
            # General args
            "--exp-name", "test",
            "--output-dir", self.path,
            "--verbose",
            # Task
            "train",
            # Tokenizer args
            "--tokenizer-type", "subword",
            "--subword-algo", "unigram",
            "--subword-train-files", train_src, train_tgt,
            "--subword-voc-size", "1000",
            # Data args
            "--train-src", train_src,
            "--train-tgt", train_tgt,
            "--valid-src", valid_src,
            "--valid-tgt", valid_tgt,
            # Model args
            "--architecture", "test_bilstm",
            # Optimization args
            "--batch-size", "2",
            "--valid-batch-size", "2",
        ])
        main.main()
        sys.argv = [sys.argv[0]]
        sys.argv.extend([
            # General args
            "--exp-name", "test",
            "--output-dir", self.path,
            "--verbose",
            # Task
            "translate",
            # Tokenizer args
            "--tokenizer-type", "subword",
            # Data args
            "--trans-src", valid_src,
            "--trans-batch-size", "2",
            # Model args
            "--architecture", "test_bilstm",
            # Translate args
            "--beam-size", "2",
            "--max-len", "3",

        ])
        main.main()

    def test_train_transformer(self):
        train_src = os.path.join(self.data_folder, "train.ja")
        train_tgt = os.path.join(self.data_folder, "train.en")
        valid_src = os.path.join(self.data_folder, "valid.ja")
        valid_tgt = os.path.join(self.data_folder, "valid.en")
        sys.argv.extend([
            # General args
            "--exp-name", "test",
            "--output-dir", self.path,
            "--verbose",
            # Task
            "train",
            # Tokenizer args
            "--tokenizer-type", "subword",
            "--subword-algo", "unigram",
            "--subword-train-files", train_src, train_tgt,
            "--subword-voc-size", "1000",
            # Data args
            "--train-src", train_src,
            "--train-tgt", train_tgt,
            "--valid-src", valid_src,
            "--valid-tgt", valid_tgt,
            # Model args
            "--architecture", "test_transformer",
            # Optimization args
            "--batch-size", "2",
            "--valid-batch-size", "2",
        ])
        main.main()

    def test_train_config_file(self):
        config_file = os.path.join(self.config_folder, "test_config.yaml")
        sys.argv.extend([
            # General args
            "--config-file", config_file,
            "--env", "train",
        ])
        main.main()

    def test_eval_ppl_config_file(self):
        config_file = os.path.join(self.config_folder, "test_config.yaml")
        sys.argv.extend([
            # General args
            "--config-file", config_file,
            "--env", "train",
        ])
        main.main()
        sys.argv = [sys.argv[0]]
        sys.argv.extend([
            # General args
            "--config-file", config_file,
            "--env", "eval_ppl",
        ])
        main.main()

    def test_translate_config_file(self):
        config_file = os.path.join(self.config_folder, "test_config.yaml")
        sys.argv.extend([
            # General args
            "--config-file", config_file,
            "--env", "train",
        ])
        main.main()
        sys.argv = [sys.argv[0]]
        sys.argv.extend([
            # General args
            "--config-file", config_file,
            "--env", "translate",
        ])
        main.main()


if __name__ == '__main__':
    unittest.main()
