import numpy as np
import torch
torch.manual_seed(0)
import os
from esimft.utils.config_file import config
from esimft.utils.processing import SuppressPrint
from esimft.utils.gearformer.sim import run_simulator, calculate_volume
from esimft.model.gearformer import GFModel
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat



if __name__ == "__main__":
    config = config()

    gfm = GFModel(config)
    gfm.encoder.eval()
    gfm.decoder.eval()
    
    data = pd.read_pickle(config.data_esimft_2)

    data_size = len(data.index)

    print("generating data...")
    dataset = []

    for i in range(0, data_size):

        print(i, " out of ", data_size)
        row = data.iloc[i]

        req_input_batch = []
        for j in range(0, config.sample_size):
            req_input = []
            for k in range(config.gf_data_req_input_start_idx, config.gf_data_req_input_end_idx+1):
                req_input.append(row.iloc[k])
            req_input_batch.append(req_input)

        seq_idx_batch, seq_batch = gfm.run(req_input_batch)

        sim_input_b = []
        for j in range(0, config.sample_size):
            sim_input_b.append({
                "id": j,
                "gear_train_sequence": seq_batch[j]
            })

        with ThreadPoolExecutor(max_workers=config.num_threads_sim) as executor, SuppressPrint():
            results = list(executor.map(run_simulator, repeat(config), sim_input_b))

        best_price_idx = -1
        best_bb_vol_idx = -1
        worst_price_idx = -1
        worst_bb_vol_idx = -1
        best_price = 1e9
        worst_price = 0
        best_bb_vol = 1e9
        worst_bb_vol = 0

        best_sol_idx = -1
        worst_sol_idx = -1
        best_value = 1e9
        worst_value = 0

        for i in range(0, len(results)):
            if results[i]["id"] == "failed":
                continue

            if config.req_name == "price":
                value = results[i]["price"]
            elif config.req_name == "bb":
                value = calculate_volume(results[i]["bounding_box_min"], results[i]["bounding_box_max"])

            if value < best_value:
                best_value = value
                best_sol_idx = i
            if value > worst_value:
                worst_value = value
                worst_sol_idx = i
                
        mid_value = np.mean([best_value, worst_value])

        dataset.append((req_input, seq_idx_batch[best_sol_idx], seq_idx_batch[worst_sol_idx], mid_value))

    val_size = int(len(dataset) * config.pref_val_ratio)
    val_dataset = dataset[:val_size]
    train_dataset = dataset[val_size:]

    print()
    print("storing files...")

    df = pd.DataFrame(train_dataset, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle(config.data_pref_train)
    df = pd.DataFrame(val_dataset, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle(config.data_pref_val)

    print(f"Preference data for new requirement, {config.req_name}, saved at: {config.data_pref_train} and {config.data_pref_val}")
