import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split


# download data
# wget https://github.com/e9t/nsmc/raw/master/ratings_train.txt -O data/nsmc/train.tsv
# wget https://github.com/e9t/nsmc/raw/master/ratings_test.txt -O data/nsmc/test.tsv


def define_config():
    p = argparse.ArgumentParser(description="Train tokenizer")
    p.add_argument("--nsmc_dir", type=str, required=True)

    config = p.parse_args()

    return config


def make_data(X, y, filename):
    with open(filename, "w") as f:
        for i, j in zip(X, y):
            label = "긍정" if j == 1 else "부정"
            f.write(f"{label}\t{i}\n")


def main():
    config = define_config()

    df_train = pd.read_csv(os.path.join(config.nsmc_dir, "train.tsv"), sep="\t")
    df_test = pd.read_csv(os.path.join(config.nsmc_dir, "test.tsv"), sep="\t")

    X_train, X_valid, y_train, y_valid = train_test_split(
        df_train["document"],  # x
        df_train["label"],  # y
        test_size=0.1,
        random_state=42,
    )

    # train data
    make_data(X_train, y_train, os.path.join(config.nsmc_dir, "train_dataset.tsv"))
    # valid data
    make_data(X_valid, y_valid, os.path.join(config.nsmc_dir, "valid_dataset.tsv"))
    # test data
    make_data(
        df_test["document"],
        df_test["label"],
        os.path.join(config.nsmc_dir, "test_dataset.tsv"),
    )


if __name__ == "__main__":
    main()
