import json
import argparse

from data.io import read_conll06_file
from model.perceptron import Perceptron
from graph.decoder import construct_and_decode
from model.parser import Parser

# CLI args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mode", type=str, required=True)
arg_parser.add_argument("--trainf", type=str, required=True)
arg_parser.add_argument("--devf", type=str, required=True)
arg_parser.add_argument("--testf", type=str, required=True)
arg_parser.add_argument("--saved_w_path", type=str, required=False, default="weights.pickle")

args = arg_parser.parse_args()

if __name__ == "__main__":
    EPOCHS = 50

    # load data
    train_set = read_conll06_file(args.trainf)
    dev_set = read_conll06_file(args.devf)
    test_set = read_conll06_file(args.testf)

    # create perceptron
    perceptron = Perceptron(is_train=True if args.mode == "train" else False)
    if args.mode == "train":
        perceptron.load(args.saved_w_path)

    # cle decoder
    decoder_fn = construct_and_decode

    # create parser
    parser = Parser(perceptron, decoder_fn)
    parser.perceptron.batchify_features(train_set)

    # train or test
    if args.mode == "train":
        parser.train(EPOCHS, train_set, dev_set)
        parser.perceptron.save(args.saved_w_path)
    else:
        # evaluate on test set
        uas_score = parser.eval(test_set)
        print(f"uas score on the test set :: {uas_score * 100}%")
