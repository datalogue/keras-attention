# Attention RNNs in Keras

Implementation and visualization of a custom Attention RNN layer in Keras for translating dates.

## Setting up the repository

0. Make sure you have Python 3.4+ installed.

1. Clone this repository to your local system

```
git clone https://github.com/datalogue/keras-attention.git
```

2. Install the requirements
(You can skip this step if you have all the requirements already installed)

```
pip install requirements.txt
```

## Creating the dataset

`cd` into `data` and run

```
python generate.py
```

This will create 4 files:
1. `training.csv` - data to train the model
2. `validation.csv` - data to evaluate the model and compare performance
3. `human_vocab.json` - vocabulary for the human dates
4. `machine_vocab.json` - vocabulary for the machine dates