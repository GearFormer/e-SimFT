import os
import numpy as np
import torch
torch.manual_seed(0)
from esimft.utils.config_file import config
from esimft.utils.processing import SuppressPrint
from esimft.utils.gearformer.sim import run_simulator
from esimft.model.gearformer import GFModel
import pandas as pd


if __name__ == "__main__":
    config = config()

    gfm = GFModel(config)

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
        req_input = []
        
        for k in range(config.gf_data_req_input_start_idx, config.gf_data_req_input_end_idx+1):
            req_input.append(row.iloc[k])
        
        if config.req_name == "speed":
            target_speed = row.iloc[config.gf_data_req_speed_idx]
        elif config.req_name == "pos":
            target_pos = [
                row.iloc[config.gf_data_req_pos_idx[0]], 
                row.iloc[config.gf_data_req_pos_idx[1]], 
                row.iloc[config.gf_data_req_pos_idx[2]],
            ]
        else:
            print("Requirement name not supported")
            exit()

        req_input_batch = []
        for j in range(0, config.BS):
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

            new_speed = sim_results["output_motion_speed"]
            new_position = sim_results["output_position"]

            if config.req_name == "speed":
                if np.isclose(new_speed, target_speed, rtol=1e-6):
                    dataset.append((req_input, seq_idx_batch[j]))
                    break
            elif config.req_name == "pos":
                if np.linalg.norm((np.array(new_position)-np.array(target_pos))) < 2e-02: 
                    dataset.append((req_input, seq_idx_batch[j]))
                    break

    val_size = int(len(dataset) * config.sft_val_ratio)
    train_dataset = dataset[val_size:]
    val_dataset = dataset[:val_size]

    train_dataset_name = config.sft_data.replace(".pkl", f"_{config.req_name}_train.pkl")
    val_dataset_name = config.sft_data.replace(".pkl", f"_{config.req_name}_val.pkl")

    print()
    print("storing files...")

    df = pd.DataFrame(train_dataset, columns=['req_input', 'chosen_seq'])
    df.to_pickle(train_dataset_name)
    df = pd.DataFrame(val_dataset, columns=['req_input', 'chosen_seq'])
    df.to_pickle(val_dataset_name)

    print(f"SFT data for {config.req_name} saved at: {train_dataset_name} and {val_dataset_name}")

