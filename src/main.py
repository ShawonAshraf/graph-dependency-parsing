import json

from data.io import read_conll06_file
from model.features import preprocess
from model.perceptron import Perceptron
from graph.decoder import construct_and_decode
from model.parser import Parser

# read configs from a json file for now
# replace later with argparse
config_file_path = "../config.json"
with open(config_file_path, "r") as json_file:
    config = json.load(json_file)

if __name__ == "__main__":
    train = read_conll06_file(config["train"])

    fdict, fs = preprocess(train)
    p = Perceptron(fdict, normalise=True)
    # p.train(1, fs)
    # p.save("./pickle.out")
    # p.load("./pickle.out")
    # print(p.weights)

    parser = Parser(p, construct_and_decode)
    heads = parser.parse(fs[0].features)
    print(heads)
