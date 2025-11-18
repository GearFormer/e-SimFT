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
    data1 = pd.read_pickle(config.sft_data)
    data2 = pd.read_pickle(config.pref_data)
    data = pd.concat([data1, data2], ignore_index=True)

    data_size = len(data.index)

    print("generating data...")
    dataset = []

    for i in range(0, data_size):

        print(i, " out of ", data_size)
        row = data.iloc[i]
        req_input_batch = []
            
        for j in range(0, config.BS):
            req_input = []
            for k in range(config.gf_data_req_input_start_idx, config.gf_data_req_input_end_idx+1):
                req_input.append(row.iloc[k])
            req_input_batch.append(req_input)


        seq_idx_batch, seq_batch = gfm.run(req_input_batch)

        for j in range(0, config.BS):
            input_data = {
                "gear_train_sequence": seq_batch[j],
                "id": 0
            }

            with SuppressPrint():
                sim_results = run_simulator(config, input_data)

            if sim_results["id"] == "failed":
                continue

            price = sim_results["price"]
            bb_min = sim_results["bounding_box_min"]
            bb_max = sim_results["bounding_box_min"]
            bb_vol = calculate_volume(bb_min, bb_max)

            price_var = np.random.uniform(0, 0.5)
            bb_vol_var = np.random.uniform(0, 0.5)

            obj_input = [price * (1+price_var), bb_vol * (1+bb_vol_var)]

            dataset.append((req_input, seq_idx_batch[j], obj_input))

        # sim_input_b = []
        # for j in range(0, config.BS):
        #     sim_input_b.append({
        #         "id": j,
        #         "gear_train_sequence": seq_batch[j]
        #     })

        # with ThreadPoolExecutor(max_workers=config.num_threads_sim) as executor, SuppressPrint():
        #     results = list(executor.map(run_simulator, repeat(config), sim_input_b))

        # for j in range(0, config.BS):

        #     if results[j]["id"] == "failed":
        #         continue

        #     price = results[j]["price"]
        #     bb_min = results[j]["bounding_box_min"]
        #     bb_max = results[j]["bounding_box_max"]
        #     bb_vol = calculate_volume(bb_min, bb_max)

        #     price_var = np.random.uniform(0, 0.5)
        #     bb_vol_var = np.random.uniform(0, 0.5)

        #     obj_input = [price * (1+price_var), bb_vol * (1+bb_vol_var)]

        #     dataset.append((req_input, seq_idx_batch[j], obj_input))

    val_size = int(len(dataset) * config.sft_val_ratio)
    train_dataset = dataset[val_size:]
    val_dataset = dataset[:val_size]

    train_dataset_name = config.sft_data.replace(".pkl", f"_new_obj_train.pkl")
    val_dataset_name = config.sft_data.replace(".pkl", f"_new_obj_val.pkl")

    print()
    print("storing files...")
    print(f"SFT data for new requirements (price and bounding box) saved at: {train_dataset_name} and {val_dataset_name}")

    df = pd.DataFrame(train_dataset, columns=['req_input', 'chosen_seq', 'new_req'])
    df.to_pickle(train_dataset_name)
    df = pd.DataFrame(val_dataset, columns=['req_input', 'chosen_seq', 'new_req'])
    df.to_pickle(val_dataset_name)
