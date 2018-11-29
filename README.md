# MT models in DyNet

## Requirements

Install with `pip install -r requirements.txt`

## Running

You can test your installation by running

```bash
# Download IWSLT data
python datasets/iwslt/download.py --year 2016 --langpair fr-en
# Test
pytest tests
```

## More info

Check [this tutorial](https://arxiv.org/abs/1703.01619) for more info on sequence to sequence model.

This project was written in [Dynet](https://github.com/clab/dynet), a cool dynamic framework for deep learning.

Hit me up at pmichel1[at]cs.cmu.edu for any question (or open an issue on github).
