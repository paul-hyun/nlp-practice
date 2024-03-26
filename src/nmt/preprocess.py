import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm


def define_config():
    p = argparse.ArgumentParser(description="Train tokenizer")
    p.add_argument("--aihub_dir", type=str, required=True)

    config = p.parse_args()

    return config


def make_data(data, fn):
    with open(fn, "w") as f:
        for row in data:
            f.write(row)
            f.write("\n")


def main():
    config = define_config()

    df_dic = {}
    for fn in tqdm(os.listdir(config.aihub_dir)):
        if fn.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(config.aihub_dir, fn))
            df = df[["원문", "번역문"]]
            df.reset_index(inplace=True)
            df_dic[fn] = df

    df_all = pd.concat(list(df_dic.values()))

    ko_train, ko_valid, en_train, en_valid = train_test_split(
        df_all["원문"],  # ko
        df_all["번역문"],  # en
        test_size=0.1,
        random_state=42,
    )

    # train data
    make_data(ko_train, os.path.join(config.aihub_dir, "train.ko"))
    make_data(en_train, os.path.join(config.aihub_dir, "train.en"))
    # valid data
    make_data(ko_valid, os.path.join(config.aihub_dir, "valid.ko"))
    make_data(en_valid, os.path.join(config.aihub_dir, "valid.en"))


if __name__ == "__main__":
    main()
