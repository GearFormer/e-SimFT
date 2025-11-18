import random
import numpy as np
import torch
torch.manual_seed(0)
import os
from esimft.utils.config_file import config
from esimft.utils.suppress_print import SuppressPrint
from esimft.utils.sim import run_simulator, calculate_volume
from esimft.model.gearformer import GFModel
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat



if __name__ == "__main__":
    config = config()

    gfm = GFModel(config)

    data = pd.read_pickle(config.pref_data)

    data_size = len(data.index)

    print("generating data...")
    dataset_price = []
    dataset_bb = []

    for i in range(0, data_size):

        print(i, " out of ", data_size)
        row = data.iloc[i]

        req_input_batch = []
        for j in range(0, config.BS):
            req_input = []
            for k in range(config.gf_data_req_start_idx, config.gf_data_req_end_idx+1):
                req_input.append(row.iloc[k])
            req_input_batch.append(req_input)

        seq_idx_batch, seq_batch = gfm.run(req_input_batch)

        sim_input_b = []
        for j in range(0, config.BS):
            sim_input_b.append({
                "id": j,
                "gear_train_sequence": seq_batch[j]
            })

        num_threads = 2
        with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
            results = list(executor.map(run_simulator, repeat(config), sim_input_b))

        best_price_idx = -1
        best_bb_vol_idx = -1
        worst_price_idx = -1
        worst_bb_vol_idx = -1
        best_price = 1e9
        worst_price = 0
        best_bb_vol = 1e9
        worst_bb_vol = 0

        for i in range(0, len(results)):
            if results[i]["id"] == "failed":
                continue

            price = results[i]["price"]
            if  price < best_price:
                best_price = price
                best_price_idx = i
            if price > worst_price:
                worst_price = price
                worst_price_idx = i

            bb_vol = calculate_volume(results[i]["bounding_box_min"], results[i]["bounding_box_max"])
            if  bb_vol < best_bb_vol:
                best_bb_vol = bb_vol
                best_bb_vol_idx = i
            if bb_vol > worst_bb_vol:
                worst_bb_vol = bb_vol
                worst_bb_vol_idx = i
                
        mid_price = np.mean([best_price, worst_price])
        mid_bb_vol = np.mean([best_bb_vol, worst_bb_vol])

        dataset_price.append((req_input, seq_idx_batch[best_price_idx], seq_idx_batch[worst_price_idx], mid_price))
        dataset_bb.append((req_input, seq_idx_batch[best_bb_vol_idx], seq_idx_batch[worst_bb_vol_idx], mid_bb_vol))

    price_val_size = int(len(dataset_price) * config.pref_val_ratio)
    val_dataset_price = dataset_price[:price_val_size]
    train_dataset_price = dataset_price[price_val_size:]
    
    bb_val_size = int(len(dataset_bb) * config.pref_val_ratio)
    val_dataset_bb = dataset_bb[:bb_val_size]
    train_dataset_bb = dataset_bb[bb_val_size:]

    print()
    print("storing files...")

    price_train_dataset_name = config.pref_data.replace(".pkl", f"_price_train.pkl")
    price_val_dataset_name = config.pref_data.replace(".pkl", f"_price_val.pkl")

    df = pd.DataFrame(train_dataset_price, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle(price_train_dataset_name)
    df = pd.DataFrame(val_dataset_price, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle(price_val_dataset_name)

    print(f"Preference data for new requirements price saved at: {price_train_dataset_name} and {price_val_dataset_name}")

    bb_train_dataset_name = config.pref_data.replace(".pkl", f"_bb_train.pkl")
    bb_val_dataset_name = config.pref_data.replace(".pkl", f"_bb_val.pkl")

    df = pd.DataFrame(train_dataset_bb, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle(bb_train_dataset_name)
    df = pd.DataFrame(val_dataset_bb, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle(bb_val_dataset_name)

    print(f"Preference data for new requirements bounding box saved at: {bb_train_dataset_name} and {bb_val_dataset_name}")
