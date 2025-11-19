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
    
    # use both data for sft and pref finetuning
    data = pd.read_pickle(config.data_esimft_1)

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

        for j in range(0, config.sample_size):
            input_data = {
                "gear_train_sequence": seq_batch[j],
                "id": 0
            }

            with SuppressPrint():
                sim_results = run_simulator(config, input_data)

            if sim_results["id"] == "failed":
                continue

            if config.req_name == "price":
                price = sim_results["price"]
                price_var = np.random.uniform(0, 0.5)
                obj_input = [price * (1+price_var)]

            elif config.req_name == "bb":               
                bb_min = sim_results["bounding_box_min"]
                bb_max = sim_results["bounding_box_max"]
                bb_vol = calculate_volume(bb_min, bb_max)
                bb_vol_var = np.random.uniform(0, 0.5)
                obj_input = [bb_vol * (1+bb_vol_var)]

            dataset.append((req_input, seq_idx_batch[j], obj_input))

    val_size = int(len(dataset) * config.sft_val_ratio)
    train_dataset = dataset[val_size:]
    val_dataset = dataset[:val_size]

    print()
    print("storing files...")

    df = pd.DataFrame(train_dataset, columns=['req_input', 'chosen_seq', 'new_req_value'])
    df.to_pickle(config.data_sft_train)
    df = pd.DataFrame(val_dataset, columns=['req_input', 'chosen_seq', 'new_req_value'])
    df.to_pickle(config.data_sft_val)

    print(f"SFT data for {config.req_name} saved at: {config.data_sft_train} and {config.data_sft_val}")