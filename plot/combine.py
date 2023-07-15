import argparse
from pathlib import Path

import pandas as pd


def main(arguments=None):
    parser = argparse.ArgumentParser("Combine two csv files for comparison")
    parser.add_argument("-f1", help="First file to combine")
    parser.add_argument("-f2", help="Second file to combine")
    parser.add_argument("-c", help="columne to combine on", type=str)

    configs = parser.parse_args(arguments)

    path1 = Path(configs.f1).stem
    path2 = Path(configs.f2).stem
    suffix_1 = f"_{path1.split('_')[0].split('-')[0]}"
    suffix_2 = f"_{path2.split('_')[0].split('-')[0]}"
    new_name = f"{path1}_vs_{path2}.csv"

    df1 = pd.read_csv(configs.f1)
    df2 = pd.read_csv(configs.f2)

    df = pd.merge(df1, df2, on="step", how="outer", suffixes=(suffix_1, suffix_2))
    df = df[["step", f"{configs.c}{suffix_1}", f"{configs.c}{suffix_2}"]]

    print(df.head())
    df.to_csv(new_name, index=False)


if __name__ == "__main__":
    main()
