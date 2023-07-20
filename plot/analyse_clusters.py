import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd


def main(arguments=None):
    parser = argparse.ArgumentParser("Combine two csv files for comparison")
    parser.add_argument("-p", help="Folder path", dest="path")

    configs = parser.parse_args(arguments)

    splits: dict = {}

    for file in Path(configs.path).glob("*cluster*.csv"):
        df = pd.read_csv(file)
        split = int(file.name.split("_")[-7].replace("split", ""))
        splits[split] = [df["inertia"].mean(), df["inertia"].std()]

    keys = []
    values = []
    for key in sorted(splits.keys(), reverse=True):
        keys.append(key)
        values.append(splits[key])
    keys = np.array(keys)
    values = np.array(values)
    print(keys)
    print(values)
    plt.plot(values[:,0], keys)
    plt.show()
        # print(f"split: {key} -> mean: {splits[key][0]}, std: {splits[key][1]}")

if __name__ == "__main__":
    main()