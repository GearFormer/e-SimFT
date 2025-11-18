import numpy as np
np.random.seed(0)
import pandas as pd
import pickle
from esimft.utils.config_file import config


if __name__ == "__main__":
    config = config()

    data = pd.read_pickle(config.pareto_test_aug_data)
    num_problems = config.pareto_exp_num_problems
    
    input_motion_type = data.iloc[:num_problems, config.gf_data_req_input_motion_type_idx].to_list()
    output_motion_type = data.iloc[:num_problems, config.gf_data_req_output_motion_type_idx].to_list()
    
    speed_ratio = data.iloc[:, config.gf_data_req_speed_idx].to_list()
    speed_min = np.min(speed_ratio)
    speed_max = np.max(speed_ratio)

    output_position_x = data.iloc[:, config.gf_data_req_pos_idx[0]]
    output_position_y = data.iloc[:, config.gf_data_req_pos_idx[1]]
    output_position_z = data.iloc[:, config.gf_data_req_pos_idx[2]]
    pos_x_min = np.min(output_position_x)
    pos_x_max = np.max(output_position_x)
    pos_y_min = np.min(output_position_y)
    pos_y_max = np.max(output_position_y)
    pos_z_min = np.min(output_position_z)
    pos_z_max = np.max(output_position_z)

    output_motion_direction = data.iloc[:num_problems, config.gf_data_req_output_motion_dir].to_list()
    output_motion_sign = data.iloc[:num_problems, config.gf_data_req_output_motion_sign].to_list()
    
    price = data.iloc[:, config.gf_data_req_price]
    price_min = np.min(price)
    price_max = np.max(price)

    bb_vol = data.iloc[:, config.gf_data_req_bb_volume]
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
    
    with open(config.pareto_problems_data, 'wb') as f:
        pickle.dump(problems, f)
