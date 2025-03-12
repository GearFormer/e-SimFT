import numpy as np
from simulator.gear_train_simulator import Simulator
from simulator.util.suppress_print import SuppressPrint
import pandas as pd
import ast
from concurrent.futures import ThreadPoolExecutor
from train_models.utils.config_file import config
from train_models.utils.data_handle import load_data

def run_simulator(input_data):

    simulator = Simulator()

    results = simulator.run(input_data)

    return results

def calculate_volume(min_corner, max_corner):
    # Unpack the corners
    min_x, min_y, min_z = min_corner
    max_x, max_y, max_z = max_corner
    
    # Calculate the dimensions
    length = max_x - min_x
    width = max_y - min_y
    height = max_z - min_z
    
    # Compute the volume
    volume = length * width * height
    return volume

if __name__ == "__main__":
    args = config()
    simulator = Simulator()
    max_length = 21
    get_dict = load_data(args)
    input_size = 8
    """
    input_[0]: input_ motion type, 1 for T and 0 for R
    input_[1]: output motion type, 1 for T and 0 for R
    input_[2]: speed ratio
    input_[3], input_[4], input_[5]: x, y, z for output position
    input_[6]: output motion vector direction xyz - 0 for x, 1 for y and 2 for z
    input_[7] : output motion vector sign 
    """

    data = pd.read_pickle("esimft_data/simft_test.pkl")
    data_size = len(data)
    print(data_size)
    # data_size = 5

    print("simulating...")
    data2sim = []

    results = []
    for i in range(0, data_size):

        row = data.iloc[i]

        req_input = []
        for k in range(2, 10):
            req_input.append(row.iloc[k])

        seq_idx = ast.literal_eval(row.iloc[-1])
        seq = list(map(get_dict.inx2name, seq_idx)) + ['<end>']
        target_inx = seq.index("<end>")
        seq = seq[:target_inx+1]

        input_data = {
            "gear_train_sequence": seq,
            "id": i
        }

        data2sim.append(input_data)

        print(i)
        results.append(run_simulator(input_data))

    # num_threads = 32
    # with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
    #     results = list(executor.map(run_simulator, data2sim))
    
    print("augmenting...")
    new_data = []
    for i in range(0, data_size):
        res = results[i]
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

    df = pd.DataFrame(new_data)
    df.to_pickle("esimft_data/simft_test_nr.pkl")
