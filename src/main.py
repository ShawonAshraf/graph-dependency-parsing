import json

# read configs from a json file for now
# replace later with argparse
config_file_path = "../config.json"
with open(config_file_path, "r") as json_file:
    config = json.load(json_file)

if __name__ == "__main__":
    print(config)
