import argparse
from pathlib import Path
from functools import reduce

import pandas as pd


def main(arguments=None):
    parser = argparse.ArgumentParser("Combine csv files for comparison")
    parser.add_argument("-p", help="Folder path", dest="path")
    parser.add_argument("-d", help="Data name", dest="data_name")
    parser.add_argument("-v", help="Value to combine", dest="combine", default="system_total_stopped")
    parser.add_argument("-f", help="Filter files", dest="filter", nargs="+", default=list())

    configs = parser.parse_args(arguments)

    data_name = configs.data_name or "avg_dist"
    df_list: list[pd.DataFrame] = []
    suffix_list: list[str] = []
    for file in Path(configs.path).glob("*conn*.csv"):
        path = file.stem
        if any(1 for f in configs.filter if path.find(f) != -1):
            continue
        suffix = f"_{path.split('_')[0].split('-')[0]}"
        if suffix.find("cql") != -1:
            split = int(file.name.split("_")[-6].replace("split", ""))
            suffix = f"{suffix}_s{split}"

        df = pd.read_csv(file)
        df_list.append(df)
        suffix_list.append(suffix)

    df = pd.DataFrame()
    df["step"] = df_list[0]["step"]
    df = reduce(lambda left, right: (pd.merge(left[0],
                                              right[0].rename(columns={c: c+right[1] for c in right[0].columns if c != 'step'}),
                                              on="step"), []),
                                              zip(df_list, suffix_list),
                                              (df, []))[0]

    value_columns = df.columns[df.columns.str.contains(f"{configs.combine}")]
    result_df = df[value_columns]
    result_df["step"] = df["step"]
    result_df[result_df.columns[::-1]].to_csv(f"cmp_{configs.combine}.csv", index=False)


if __name__ == "__main__":
    main()
