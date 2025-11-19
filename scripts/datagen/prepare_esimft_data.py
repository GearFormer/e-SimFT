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
    simft_test_idx = test_size // 2
    pareto_test_idx = test_size
    esimft_split_idx = int(test_size * 2 + (data_size - test_size * 2) / 2)

    simft_test = shuffled_data[:simft_test_idx]
    pareto_test = shuffled_data[simft_test_idx:pareto_test_idx]
    esimft_1 = shuffled_data[pareto_test_idx:esimft_split_idx]
    esimft_2 = shuffled_data[esimft_split_idx:]

    print("\nSplit sizes:")
    print(f"  eSimFT data split 1 size:    {len(esimft_1)} rows")
    print(f"  eSimFT data split 2 size:   {len(esimft_2)} rows")
    print(f"  SimFT test size:   {len(simft_test)} rows")
    print(f"  Pareto test size:  {len(pareto_test)} rows")
    print(f"  Total check:       {len(simft_test) + len(pareto_test) + len(esimft_1) + len(esimft_2)} rows")

    print(f"\nStoring pickle files")
    esimft_1.to_pickle(config.data_esimft_1)
    esimft_2.to_pickle(config.data_esimft_2)
    simft_test.to_pickle(config.data_simft_test)
    pareto_test.to_pickle(config.data_pareto_test)

    print("Saved files:")
    print(f"  eSimFT split 1   -> {config.data_esimft_1}")
    print(f"  eSimFT split 2  -> {config.data_esimft_2}")
    print(f"  SimFT test  -> {config.data_simft_test}")
    print(f"  Pareto test -> {config.data_pareto_test}")

if __name__ == "__main__":
    config = config()

    main(config)
