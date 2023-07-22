import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def flatten(df: pd.DataFrame):
    flat_cols = []
    for i in df.columns:
        flat_cols.append(i[0] + "_" + i[1])
    df.columns = flat_cols

    return df


def main(arguments=None):
    parser = argparse.ArgumentParser("Combine two csv files for comparison")
    parser.add_argument("-p", help="Folder path", dest="path")
    parser.add_argument("-d", help="Data name", dest="data_name")
    parser.add_argument("-std", help="Show variance", dest="std", action="store_true")
    parser.add_argument("-a", help="Agent id", dest="agent_ids", nargs="+")
    parser.add_argument("-mav", help="Moving avg size", dest="mav", type=int, default=50)

    configs = parser.parse_args(arguments)

    fig, ax = plt.subplots()
    data_name = configs.data_name or "avg_dist"
    for file in Path(configs.path).glob("*cluster*.csv"):
        df = pd.read_csv(file)
        print(df.head())
        print()
        split = int(file.name.split("_")[-7].replace("split", ""))
        df = df.groupby(["step", "agent_id"]).agg(["mean", "std"])
        df = flatten(df).reset_index().set_index("step")
        if configs.agent_ids is not None:
            df = df[df["agent_id"].isin([int(id) for id in configs.agent_ids])]
        df["value_mean"] = df[f"{data_name}_mean"].rolling(configs.mav).mean()
        df["value_std"] = df[f"{data_name}_std"].rolling(configs.mav).mean()
        ax.plot(df.index, df["value_mean"], label=f"split: {split}")
        if configs.std:
            ax.fill_between(df.index, df["value_mean"] - df["value_std"], df["value_mean"] + df["value_std"], alpha=0.2)
    ax.legend()
    ax.set_xlabel("steps")
    ax.set_title(f"{data_name} over steps")
    plt.show()


if __name__ == "__main__":
    main()
