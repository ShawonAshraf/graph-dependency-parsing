import json

from data.io import read_conll06_file
from model.perceptron import Perceptron
from graph.decoder import construct_and_decode
from model.parser import Parser

from sklearn.metrics import accuracy_score

# read configs from a json file for now
# replace later with argparse
config_file_path = "../config.json"
with open(config_file_path, "r") as json_file:
    config = json.load(json_file)

if __name__ == "__main__":
    train = read_conll06_file(config["train"])
    dev = read_conll06_file(config["dev"])
    test = read_conll06_file(config["test"])

    perceptron = Perceptron()
    decoder_fn = construct_and_decode


    parser = Parser(perceptron, decoder_fn)
    parser.perceptron.batchify_features(train)

