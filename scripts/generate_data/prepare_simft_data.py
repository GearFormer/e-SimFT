import argparse
from pathlib import Path
from esimft.utils.config_file import config

import pandas as pd


def main(config):

    # Load and concatenate all CSVs
    dataframes = []
    csv_files = [config.gearformer_val_data, config.gearformer_test_data]
    for csv_path in csv_files:
        print(f"Loading {csv_path} ...")
        df = pd.read_csv(csv_path, header=None)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)
    data_size = len(data.index)
    print(f"Total concatenated rows: {data_size}")

    # Shuffle data
    shuffled_data = data.sample(frac=1).reset_index(drop=True)

    # Compute split indices
    test_size = int(data_size * config.simft_test_ratio)
    simft_test_idx = test_size
    pareto_test_idx = test_size * 2
    simft_train_idx = int(test_size * 2 + (data_size - test_size * 2) / 2)

    simft_test = shuffled_data[:simft_test_idx]
    pareto_test = shuffled_data[simft_test_idx:pareto_test_idx]
    sft_train = shuffled_data[pareto_test_idx:simft_train_idx]
    pref_train = shuffled_data[simft_train_idx:]

    print("\nSplit sizes:")
    print(f"  SimFT test size:   {len(simft_test)} rows")
    print(f"  Pareto test size:  {len(pareto_test)} rows")
    print(f"  SFT train size:    {len(sft_train)} rows")
    print(f"  Pref train size:   {len(pref_train)} rows")
    print(f"  Total check:       {len(simft_test) + len(pareto_test) + len(sft_train) + len(pref_train)} rows")

    print(f"\nStoring pickle files")
    sft_train.to_pickle(config.sft_data)
    pref_train.to_pickle(config.pref_data)
    simft_test.to_pickle(config.simft_test_data)
    pareto_test.to_pickle(config.pareto_test_data)

    print("Saved files:")
    print(f"  SFT train   -> {config.sft_data}")
    print(f"  Pref train  -> {config.pref_data}")
    print(f"  SimFT test  -> {config.simft_test_data}")
    print(f"  Pareto test -> {config.pareto_test_data}")

if __name__ == "__main__":
    config = config()

    main(config)
