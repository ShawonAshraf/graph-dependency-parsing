import json

from data.io import read_conll06_file
from model.eval import las, uas
from model.features import preprocess, create_vector_representation, encode_labels

# read configs from a json file for now
# replace later with argparse
config_file_path = "../config.json"
with open(config_file_path, "r") as json_file:
    config = json.load(json_file)

if __name__ == "__main__":
    # sentences = read_conll06_file(config["file"])
    # write to file
    # out_file_path = "../out.conll06"
    # write_conll06_file(sentences, out_file_path)
    gold = read_conll06_file(config["eval_gold"])
    pred = read_conll06_file(config["eval_pred"])

    train = read_conll06_file(config["train"])

    uas_score = uas(gold, pred)
    print(uas_score)

    las_score = las(gold, pred)
    print(las_score)

    fdict, fs = preprocess(train)
    # print(create_vector_representation(fdict, fs))
    print(fs[1])

    rep = create_vector_representation(fdict, fs)
    lab = encode_labels(fs)
    # print(rep[0])

    # print(fdict)
    # print(gold[23])
    # extract_feature(gold[23])
