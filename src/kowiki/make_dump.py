import os
import argparse
import json
import re


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--kowiki_home", required=True)
    p.add_argument("--n_files", type=int, default=-1)

    config = p.parse_args()

    return config


def main(config):
    files = []
    for name in os.listdir(config.kowiki_home):
        path = os.path.join(config.kowiki_home, name)
        if os.path.isdir(path):
            for name in os.listdir(path):
                if re.match(r"wiki_[0-9]{2}", name):
                    files.append(os.path.join(path, name))

    with open(os.path.join(config.kowiki_home, "wiki_dump.txt"), "w") as f_out:
        for i, file in enumerate(sorted(files)):
            if config.n_files > 0 and i >= config.n_files:
                break
            with open(file) as f_in:
                for line in f_in:
                    text = json.loads(line)["text"]
                    f_out.write(text)
                    f_out.write("\n" * 4)


if __name__ == "__main__":
    config = define_config()
    main(config)
