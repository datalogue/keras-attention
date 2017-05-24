import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from models.NMT import simpleNMT
from utils.examples import run_example
from data.reader import Vocabulary


def load_examples(file_name):
    with open(file_name) as f:
        return [s.replace('\n', '') for s in f.readlines()]

# create a directory if it doesn't already exist
if not os.path.exists('./attention_maps/'):
    os.makedirs('./attention_maps/')


class Visualizer(object):

    def __init__(self, padding=None):
        """
            Visualizes attention maps
            :param padding: the padding to use for the sequences.
        """
        self.padding = padding
        self.input_vocab = Vocabulary(
            './data/human_vocab.json', padding=padding)
        self.output_vocab = Vocabulary(
            './data/machine_vocab.json', padding=padding)

    def set_models(self, pred_model, proba_model):
        """
            Sets the models to use
            :param pred_model: the prediction model
            :param proba_model: the model that outputs the activation maps
        """
        self.pred_model = pred_model
        self.proba_model = proba_model

    def attention_map(self, text):
        """
            Text to visualze attention map for.
        """
        # encode the string
        d = self.input_vocab.string_to_int(text)

        # get the output sequence
        predicted_text = run_example(
            self.pred_model, self.input_vocab, self.output_vocab, text)

        # get the lengths of the string
        input_length = len(text)+5
        # get the activation map
        activation_map = np.squeeze(self.proba_model.predict(np.array([d])))[
            0:input_length, 0:input_length]

        plt.clf()
        plt.imshow(activation_map, interpolation='nearest', cmap='gray')
        plt.yticks(range(input_length), predicted_text[:input_length])
        plt.xticks(range(input_length), text)
        plt.xlabel('Input Sequence')
        plt.ylabel('Output Sequence')
        plt.savefig('./attention_maps/'+text.replace('/', '')+'.pdf')


def main(examples, args):
    print('Total Number of Examples:', len(examples))
    weights_file = os.path.expanduser(args.weights)
    print('Weights loading from:', weights_file)
    viz = Visualizer(padding=args.padding)
    print('Loading models')
    pred_model = simpleNMT(trainable=False,
                           pad_length=args.padding,
                           n_chars=viz.input_vocab.size(),
                           n_labels=viz.output_vocab.size())

    pred_model.load_weights(weights_file, by_name=True)
    pred_model.compile(optimizer='adam', loss='categorical_crossentropy')

    proba_model = simpleNMT(trainable=False,
                            pad_length=args.padding,
                            n_chars=viz.input_vocab.size(),
                            n_labels=viz.output_vocab.size(),
                            return_probabilities=True)

    proba_model.load_weights(weights_file, by_name=True)
    proba_model.compile(optimizer='adam', loss='categorical_crossentropy')

    viz.set_models(pred_model, proba_model)

    print('Models loaded')

    for example in examples:
        viz.attention_map(example)

    print('Completed visualizations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-w', '--weights', metavar='|',
                            help="""Location of weights""",
                            required=True)
    named_args.add_argument('-e', '--examples', metavar='|',
                            help="""Example string/file to visualize attention map for
                                    If file, it must end with '.txt'""",
                            required=True)
    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Length of padding""",
                            required=False, default=50, type=int)
    args = parser.parse_args()

    if '.txt' in args.examples:
        examples = load_examples(args.examples)
    else:
        examples = [args.examples]

    main(examples, args)
