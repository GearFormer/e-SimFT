import numpy as np
import torch
torch.manual_seed(0)
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
    
    data = pd.read_pickle(config.ric_aug_data)

    data_size = len(data.index)

    print("generating data...")
    dataset = []

    for i in range(0, data_size):

        print(i, " out of ", data_size)
        row = data.iloc[i]

        req_input = []
        for k in range(config.gf_data_req_input_start_idx, config.gf_data_req_input_end_idx+1):
            req_input.append(row.iloc[k])
        req_input_batch = config.sample_size * [req_input]

        target_speed = req_input[config.gf_data_req_speed_idx]
        target_pos = [req_input[config.gf_data_req_pos_idx[0]], req_input[config.gf_data_req_pos_idx[1]], req_input[config.gf_data_req_pos_idx[2]]]

        seq_idx_batch, seq_batch = gfm.run(req_input_batch)

        sim_input_b = []
        for j in range(0, config.sample_size):
            sim_input_b.append({
                "id": j,
                "gear_train_sequence": seq_batch[j]
            })

        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
            results = list(executor.map(run_simulator, repeat(config), sim_input_b))

        for i in range(0, len(results)):
            if results[i]["id"] == "failed":
                continue

            w = [0, 0, 0, 0]

            speed = results[i]["output_motion_speed"]
            pos = results[i]["output_position"]
            price = results[i]["price"]
            bb_vol = calculate_volume(results[i]["bounding_box_min"], results[i]["bounding_box_max"])

            if np.isclose(speed, target_speed, rtol=1e-6):
                w[0] = 1

            if np.linalg.norm((np.array(pos)-np.array(target_pos))) < 2e-02: 
                w[1] = 1

            price_var = np.random.uniform(-0.5, 0.5)
            target_price = price * (1+price_var)
            if price_var > 0:
                w[2] = 1
            
            bb_vol_var = np.random.uniform(-0.5, 0.5)
            target_bb_vol = bb_vol * (1+bb_vol_var)
            if bb_vol_var > 0:
                w[3] = 1

            new_req = [target_price, target_bb_vol]

            dataset.append((req_input, seq_idx_batch[i], new_req, w))

    val_size = int(len(dataset) * config.sft_val_ratio)
    val_dataset = dataset[:val_size]
    train_dataset = dataset[val_size:]

    print()
    print("storing files...")

    df = pd.DataFrame(train_dataset, columns=['req_input', 'seq', 'new_req', 'weights'])
    df.to_pickle(config.ric_train_data)
    df = pd.DataFrame(val_dataset, columns=['req_input', 'seq', 'new_req', 'weights'])
    df.to_pickle(config.ric_val_data)
