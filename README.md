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

More specifically given an input sentence 

![in](https://www.latex4technics.com/imgtemp/vco293-3.png?1493055092)

and an output sentence 

![out](https://www.latex4technics.com/imgtemp/j213d5-1.png?1493055162)

The model is trained to minimize the conditional log-likelihood

![ll](https://www.latex4technics.com/imgtemp/owrs1n-1.png?1493055660)

Where the probability is computed using an encoder decoder network :

1. Encoder: with the default parameter, this encodes the source sentence with a bidirectional LSTM

![encode](https://www.latex4technics.com/imgtemp/cx475y-1.png?1493056081)

2. Attention : the attention scores are computed based on the encodings and the previous decoder output.

![attention](https://www.latex4technics.com/imgtemp/ttii7g-1.png?1493055518)

3. decoding : this uses an LSTM as well.

![decode](https://www.latex4technics.com/imgtemp/ykovgk-1.png?1493056131)

## Performance

With the configuration stored in `config/best_config.yaml`, a BLEU score of 27.26 is attained on the test set.

Here are some samples (randomly selected, edited some spacing for clarity):

| Target | Hypothesis |
|--------|------------|
|`i didn't mention the skin of my beloved fish, which was delicious -- and i don't like fish skin ; i don't like it seared, i don't like it crispy. ` | `i don't mention the skin of my beloved fish that was delicious, and i don't like a UNK. i don't like it. i don't like you.`|
|`we will be as good at whatever we do as the greatest people in the world. | we 're going to be so good at doing whatever the most significant people in the world.|
|`i actually am. | that 's me even.|
|`tremendously challenging. | a great challenge.|
|`if you 're counting on it for 100 percent, you need an incredible miracle battery. | if you want to support 100 percent of it, you need an incredible UNK.|

## Known issue

As of now you can't configure the global dynet parameters with the yaml config, you need to specify them manually in the command line
