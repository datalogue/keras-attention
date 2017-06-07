# Attention RNNs in Keras

Implementation and visualization of a custom RNN layer with attention in Keras for translating dates.

## Setting up the repository

0. Make sure you have Python 3.4+ installed.

1. Clone this repository to your local system

```
git clone https://github.com/datalogue/keras-attention.git
```

2. Install the requirements
(You can skip this step if you have all the requirements already installed)

We would recommend using GPU's otherwise training might be prohbitively slow:

```
pip install -r requirements-gpu.txt
```

If you do not have a GPU or want to prototype on your local machine:

```
pip install -r requirements.txt
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


## Running the model

We highly recommending having a machine with a GPU to run this software, otherwise training might be prohibitively slow. To see what arguments are accepted you can run `python run.py -h` from the main directory:

```
usage: run.py [-h] [-e |] [-g |] [-p |] [-t |] [-v |] [-b |]

optional arguments:
  -h, --help            show this help message and exit

named arguments:
  -e |, --epochs |      Number of Epochs to Run
  -g |, --gpu |         GPU to use
  -p |, --padding |     Amount of padding to use
  -t |, --training-data |
                        Location of training data
  -v |, --validation-data |
                        Location of validation data
  -b |, --batch-size |  Location of validation data
```

All parameters have default values, so if you want to just run it, you can type `python run.py`. You can always stop running the model early using `Ctrl+C`.

## Visualizing Attention

You can use the script `visualize.py` to visualize the data. Run with the `-h` argument to see what is accepted:

```
usage: visualize.py [-h] -w | -e | [-p |]

optional arguments:
  -h, --help          show this help message and exit

named arguments:
  -w |, --weights |   Location of weights
  -e |, --examples |  Example string/file to visualize attention map for If
                      file, it must end with '.txt'
  -p |, --padding |   Length of padding
```

The default `padding` parameters correspond between `run.py` and `visualize.py` and therefore, if you change this make sure to note it. You must supply the path to the weights you want to use and an example/file of examples. An example file is provided in `utils/examples.txt`. 

### Example visualizations

Here are some example visuals you can obtain:

![image](https://user-images.githubusercontent.com/6295292/26899949-bbac0c7c-4b9e-11e7-84d6-c2f31166af07.png)

*The model has learned that “Saturday” has no predictive value!*

![image](https://user-images.githubusercontent.com/6295292/26899993-dd40e416-4b9e-11e7-99ec-71d536832347.png)

*We can see the weirdly formatted date “January 2016 5” is incorrectly translated as 2016–01–02 where the “02” comes from the “20” in 2016*

### Help

Start an issue if you find a bug or would like to contribute!

For other matters, you can contact [@zafarali](http://www.github.com/zafarali) at zaf@datalogue.io or us directly contact@datalogue.io 


### Acknowledgements

As with all open source code, we could not have built this without other code out there. Special thanks to:

1. [rasmusbergpalm/normalization](https://github.com/rasmusbergpalm/normalization/blob/master/babel_data.py) - for some of the data generation code.
2. [joke2k/faker](https://github.com/joke2k/faker) for their fake data generator.

### References

Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. 
["Neural machine translation by jointly learning to align and translate." 
arXiv preprint arXiv:1409.0473 (2014).](https://arxiv.org/abs/1409.0473)