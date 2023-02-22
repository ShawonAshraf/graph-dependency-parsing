import json

from data.io import read_conll06_file
from model.features import preprocess
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

    fdict, fs = preprocess(train)
    fdcir2, fs2 = preprocess(dev)
    p = Perceptron(fdict, normalise=True, train=False)
    # p.train(1, fs)
    # p.save("./pickle.out")
    p.load("./pickletrain.out")
    # print(p.weights)

    parser = Parser(p, construct_and_decode)
    heads = parser.parse(fs[0].features)

    gold_heads = [l[1] for l in fs[0].labels]

    print(accuracy_score(y_true=gold_heads, y_pred=heads))

    # parser.train(epochs=5, train_instances=fs, dev_instances=fs2)
    # parser.perceptron.save("./pickletrain.out")
