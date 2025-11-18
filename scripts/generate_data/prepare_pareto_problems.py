import random
random.seed(0)
import numpy as np
np.random.seed(0)
import pandas as pd
import pickle


if __name__ == "__main__":

    data = pd.read_pickle("esimft_data/pareto_test_obj.pkl")

    num_problems = 30
    
    input_motion_type = data.iloc[:num_problems, 2].to_list()
    output_motion_type = data.iloc[:num_problems, 3].to_list()
    
    speed_ratio = data.iloc[:, 4].to_list()
    speed_min = np.min(speed_ratio)
    speed_max = np.max(speed_ratio)

    output_position_x = data.iloc[:, 5]
    output_position_y = data.iloc[:, 6]
    output_position_z = data.iloc[:, 7]
    pos_x_min = np.min(output_position_x)
    pos_x_max = np.max(output_position_x)
    pos_y_min = np.min(output_position_y)
    pos_y_max = np.max(output_position_y)
    pos_z_min = np.min(output_position_z)
    pos_z_max = np.max(output_position_z)

    output_motion_direction = data.iloc[:num_problems, 8].to_list()
    output_motion_sign = data.iloc[:num_problems, 9].to_list()
    
    price = data.iloc[:, 11]
    price_min = np.min(price)
    price_max = np.max(price)

    bb_vol = data.iloc[:, 12]
    bb_vol_min = np.min(bb_vol)
    bb_vol_max = np.max(bb_vol)

    # generate problems
    speeds = np.random.uniform(speed_min, speed_max, num_problems)
    pos_x = np.random.uniform(pos_x_min, pos_x_max, num_problems)
    pos_y = np.random.uniform(pos_y_min, pos_y_max, num_problems)
    pos_z = np.random.uniform(pos_z_min, pos_z_max, num_problems)
    prices = np.random.uniform(price_min, price_max, num_problems)
    bb_vols = np.random.uniform(bb_vol_min, bb_vol_max, num_problems)

    problems = np.array((input_motion_type, output_motion_type, speeds, pos_x, pos_y, pos_z, 
                        output_motion_direction, output_motion_sign, prices, bb_vols)).T
    
    with open('esimft_data/pareto_problems.pkl', 'wb') as f:
        pickle.dump(problems, f)
