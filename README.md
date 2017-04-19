# Attentional Seq2Sqeq model in DyNet

## Requirements

You will need [dynet](https://github.com/clab/dynet.git) to run this code. See the installation instructions [here](http://dynet.readthedocs.io/en/latest/python.html)

On top of this, you will also require

    numpy
    argparse
    pyyaml

## Running

You can test your installation by running

    python run.py -c config/test_config.yaml -e train

The `--config_file/-c` argument specifies the path to a `.yaml` file containing the options, see `config/test_config.yaml` for an example.

You can use the `--env/-e` argument to specify a subset of parameters you wish to use (for instance to differentiate between training and testing). Again, see `config/test_config.yaml` for an example.

Alternatively, you can set all the parameters via the command line, run 

    python run.py -h

For more details on the available parameters

## Data

The data is from the IWSLT2016 workshop for German-English translation, separated into a training set, validation set (the dev2010 set from IWSLT), and test set (the tst2010 set). There is an additional blind test set for which translations are not provided.

## Model

The current model uses LSTMs (`dynet.VanillaLSTM`) for the encoder(s) and decoder, as well as MLP attention.

## Performance

With the configuration stored in `config/best_config.yaml`, a BLEU score of 25.81 is attained on the test set

## Known issue

As of now you can't configure the global dynet parameters with the yaml config, you need to specify them manually in the command line
