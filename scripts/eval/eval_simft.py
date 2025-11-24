import numpy as np
import os
import torch
from esimft.utils.config_file import config
from esimft.model.gearformer import GFModel, ObjEncoder
from esimft.model.gearformer_simft import GearFormerSimFT
from esimft.utils.data_handle import DataHandler
from esimft.utils.gearformer.sim import run_simulator, calculate_volume
from esimft.utils.processing import SuppressPrint
from itertools import repeat
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import root_mean_squared_log_error
import math
from tqdm import tqdm


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config()
    data_handler = DataHandler(config)

    if config.req_name in ["bb", "price"]:
        test_data = pd.read_pickle(config.data_simft_test_aug)
    else:
        test_data = pd.read_pickle(config.data_simft_test)
        
    data_size = len(test_data.index)
    
    orig_req_b = []
    target_objs = []

    encoder_path = torch.load(os.path.join(config.checkpoint_path, config.gearformer_encoder_checkpoint_name))
    decoder_path = torch.load(os.path.join(config.checkpoint_path, config.gearformer_decoder_checkpoint_name))

    gfm = GFModel(config, device)
    encoder = gfm.encoder
    decoder = gfm.decoder

    if config.sft_mode == "baseline":
        sft_model = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
    elif config.sft_mode == "original_req":
        sft_model = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
        sft_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.sft_model_checkpoint_name), map_location=device))
    elif config.sft_mode == "new_req":
        new_req_encoder = ObjEncoder(input_size=1, output_size=config.dim).to(device)
        sft_model = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
        sft_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.sft_model_checkpoint_name), map_location=device))

    orig_req = []
    new_req = []

    for i in range(0, data_size):
        row = test_data.iloc[i]

        input_motion_type = row.iloc[2]
        output_motion_type = row.iloc[3]
        speed_ratio = row.iloc[4]
        output_position = np.array([row.iloc[5], row.iloc[6], row.iloc[7]])
        output_motion_direction = row.iloc[8]
        output_motion_sign = row.iloc[9]

        orig_req.append([input_motion_type, output_motion_type, speed_ratio, output_position[0], output_position[1], output_position[2],
                    output_motion_direction, output_motion_sign])

        if config.req_name == "bb":
            bb_vol = row.iloc[12]
            new_req.append([bb_vol*0.85])
        elif config.req_name == "price":
            price = row.iloc[11]
            new_req.append([price*0.9])

    speed_msle = 0.0
    pos_euc = 0.0
    price_req_met = 0
    bb_req_met = 0
    valid = 0

    for start in tqdm(range(0, data_size, config.BS)):
        end = min(start + config.BS, data_size)

        cur_bs = end - start
        if cur_bs == 0:
            continue

        orig_req_inputs = torch.tensor(orig_req[start:end], dtype=torch.float32, device=device)

        if config.sft_mode == "new_req":
            new_req_inputs = torch.tensor(new_req[start:end], dtype=torch.float32, device=device)
            inputs = (orig_req_inputs, new_req_inputs)

        else:
            inputs = (orig_req_inputs, )

        start_token = 0
        prompts = torch.full(
            (cur_bs, 1),
            fill_value=start_token,
            dtype=torch.long,
            device=device,
        )

        # ---- Generate sequences for this batch ----
        pred_batch = sft_model.generate(inputs, prompts)

        out_seq_batch = []
        for pred in pred_batch:
            out_seq = ["<start>"] + list(map(data_handler.inx2name, pred.cpu().tolist())) + ['<end>']
            target_inx = out_seq.index("<end>")
            out_seq = out_seq[:target_inx+1]
            out_seq_batch.append(out_seq)

        # ---- Prepare simulator inputs for this batch ----
        sim_input_b = []
        for idx, i in enumerate(range(start, end)):
            sim_input_b.append({
                "id": i,  # keep global id so metrics align
                "gear_train_sequence": out_seq_batch[idx],
            })

        # ---- Run simulator in parallel for this batch ----
        with ThreadPoolExecutor(max_workers=config.num_threads_sim) as executor, SuppressPrint():
            results_batch = list(executor.map(run_simulator, repeat(config), sim_input_b))

        # ---- Accumulate metrics for this batch ----
        for local_idx, result in enumerate(results_batch):
            i = start + local_idx  # global index

            if result["id"] == "failed":
                continue

            elif config.req_name == "speed":
                speed_msle += root_mean_squared_log_error(
                    [orig_req[i][2]],
                    [result["output_motion_speed"]],
                )

            elif config.req_name == "pos":
                gt_pos = [orig_req[i][3], orig_req[i][4], orig_req[i][5]]
                pos_euc += math.dist(result["output_position"], gt_pos)

            elif config.req_name == "bb":
                actual_bb_min = result["bounding_box_min"]
                actual_bb_max = result["bounding_box_max"]
                actual_bb_vol = calculate_volume(actual_bb_min, actual_bb_max)
                target_bb_vol = new_req[i][0]
                if actual_bb_vol <= target_bb_vol:
                    bb_req_met += 1

            elif config.req_name == "price":
                actual_price = result["price"]
                target_price = new_req[i][0]
                if actual_price <= target_price:
                    price_req_met += 1

            else:
                raise ValueError(f"Unknown req_name: {config.req_name}")

            valid += 1

    print("computing metrics... ")

    
    print("output validity", valid/data_size)

    if config.req_name == "speed":
        print("average speed error", speed_msle/valid)

    elif config.req_name == "pos":
        print("average position error", pos_euc/valid)

    elif config.req_name == "price":
        print("price requirement met", price_req_met/valid)

    elif config.req_name == "bb":
        print("bounding box requirement met", bb_req_met/valid)

