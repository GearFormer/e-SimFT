import numpy as np
import pandas as pd
import ast
from concurrent.futures import ThreadPoolExecutor
from esimft.utils.config_file import config
from esimft.utils.data_handle import DataHandler
from esimft.utils.gearformer.sim import run_simulator, calculate_volume


if __name__ == "__main__":
    config = config()
    data_handler = DataHandler(config)

    if config.aug_data_type == "pref":
        data = pd.read_pickle(config.pref_data)
        save_file = config.ric_aug_data
    elif config.aug_data_type == "pareto_test":
        data = pd.read_pickle(config.pareto_test_data)
        save_file = config.pareto_test_aug_data
    elif config.aug_data_type == "simft_test":
        data = pd.read_pickle(config.simft_test_data)
        save_file = config.simft_test_aug_data
    else:
        print("aug_data_type not specified / supported")
        exit()

    data_size = len(data)

    print("simulating...")

    results = []
    for i in range(0, data_size):

        print(i, " out of ", data_size)

        row = data.iloc[i]

        req_input = []
        for k in range(config.gf_data_req_input_start_idx, config.gf_data_req_input_end_idx+1):
            req_input.append(row.iloc[k])

        seq_idx = ast.literal_eval(row.iloc[-1])
        seq = list(map(data_handler.inx2name, seq_idx)) + ['<end>']
        target_inx = seq.index("<end>")
        seq = seq[:target_inx+1]

        input_data = {
            "gear_train_sequence": seq,
            "id": i
        }
        
        results.append(run_simulator(config, input_data))

    print("augmenting...")
    new_data = []
    for i in range(0, data_size):
        res = results[i]
        if res["id"] == "failed":
            continue
            
        price = res["price"]
        bb_min = res["bounding_box_min"]
        bb_max = res["bounding_box_max"]

        row = data.iloc[i]
        a = row.iloc[0]
        b = row.iloc[1]
        input_motion_type = row.iloc[2]
        output_motion_type = row.iloc[3]
        speed_ratio = row.iloc[4]
        output_position = np.array([row.iloc[5], row.iloc[6], row.iloc[7]])
        output_motion_direction = row.iloc[8]
        output_motion_sign = row.iloc[9]
        seq = row.iloc[10]

        new_data.append([row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5],
                         row.iloc[6], row.iloc[7], row.iloc[8], row.iloc[9], row.iloc[10],
                         price, calculate_volume(bb_min, bb_max)])

    print("storing files...")
    df = pd.DataFrame(new_data)
    save_file = config.ric_aug_data
    df.to_pickle(save_file)
    print(f"Augmented new requirement data for {config.aug_data_type} benchmarking saved at: {save_file}")
