import random
random.seed(0)
import numpy as np
np.random.seed(0)
import pandas as pd
import pickle
import argparse


def prepare_req(data, N):

    req_inputs= {
        "default2": [],
        "default3": [],
        "speed_pos": {
            "speed_eps": [],
            "pos_eps": []
        },
        "speed_price": {
            "speed_eps": [],
            "price_eps": []
        },
        "speed_bb": {
            "speed_eps": [],
            "bb_eps": []
        },
        "pos_price": {
            "pos_eps": [],
            "price_eps": []
        },
        "pos_bb": {
            "pos_eps": [],
            "bb_eps": []
        },            
        "price_bb": {
            "price_eps": [],
            "bb_eps": []
        },
        "speed_pos_bb": {
            "speed_eps": [],
            "pos_eps": [],
            "bb_eps": []
        },
        "speed_pos_price": {
            "speed_eps": [],
            "pos_eps": [],
            "price_eps": []
        },
        "speed_bb_price": {
            "speed_eps": [],
            "bb_eps": [],
            "price_eps": []
        },            
        "pos_price_bb": {
            "pos_eps": [],
            "price_eps": [],
            "bb_eps": []
        }
    }
    N2 = N

    for i in range(len(data)):
        row = data[i]
        input_motion_type = row[0]
        output_motion_type = row[1]
        speed_ratio = row[2]
        output_position = np.array([row[3], row[4], row[5]])
        output_motion_direction = row[6]
        output_motion_sign = row[7]
        price = row[8]
        bb_vol = row[9]

        req_input = [input_motion_type, output_motion_type, speed_ratio, output_position[0], output_position[1], 
                        output_position[2], output_motion_direction, output_motion_sign, price, bb_vol]
                
        default2 = []
        default3 = []

        two_speed1 = []
        two_pos1 = []
        two_price1 = []
        two_pos2 = []
        two_price2 = []
        two_bb2 = []
        
        # for two-req problems

        for j in range(0, N):
            default2.append(req_input)

        var_orig = np.linspace(0.5, 1.5, int(N/2))
        var_new = np.linspace(1.0, 2.0, int(N/2))

        for j in range(0, int(N/2)):

            r = req_input.copy()
            r[2] *= var_orig[j]
            two_speed1.append(r)

            r = req_input.copy()
            r[3] *= var_orig[j]
            r[4] *= var_orig[j]
            r[5] *= var_orig[j]
            two_pos1.append(r)
            two_pos2.append(r)

            r = req_input.copy()
            r[8] *= var_new[j]
            two_price1.append(r)
            two_price2.append(r)

            r = req_input.copy()
            r[9] *= var_new[j]       
            two_bb2.append(r) 

        three_speed1 = []
        three_pos1 = []
        three_pos2 = []
        three_bb2 = []
        three_price2 = []
        three_bb3 = []
        three_price3 = []

        # for three-req problems
        for j in range(0, N2):
            default3.append(req_input)

        var_orig = np.linspace(0.5, 1.5, int(N2/3))
        var_new = np.linspace(1.0, 2.0, int(N2/3))

        for j in range(0, int(N2/3)):

            r = req_input.copy()
            r[2] *= var_orig[j]
            three_speed1.append(r)
            
            r = req_input.copy()
            r[3] *= var_orig[j]
            r[4] *= var_orig[j]
            r[5] *= var_orig[j]
            three_pos1.append(r)
            three_pos2.append(r)
                
            r = req_input.copy()
            r[9] *= var_new[j]
            three_bb2.append(r)
            three_bb3.append(r)

            r = req_input.copy()   
            r[8] *= var_new[j]             
            three_price2.append(r)
            three_price3.append(r) 

        req_inputs["default2"].append(default2)
        req_inputs["default3"].append(default3)
        req_inputs["speed_pos"]["speed_eps"].append(two_speed1)
        req_inputs["speed_pos"]["pos_eps"].append(two_pos2)
        req_inputs["speed_price"]["speed_eps"].append(two_speed1)
        req_inputs["speed_price"]["price_eps"].append(two_price2)
        req_inputs["speed_bb"]["speed_eps"].append(two_speed1)
        req_inputs["speed_bb"]["bb_eps"].append(two_bb2)
        req_inputs["pos_price"]["pos_eps"].append(two_pos1)
        req_inputs["pos_price"]["price_eps"].append(two_price2)
        req_inputs["pos_bb"]["pos_eps"].append(two_pos1)
        req_inputs["pos_bb"]["bb_eps"].append(two_bb2)
        req_inputs["price_bb"]["price_eps"].append(two_price1)
        req_inputs["price_bb"]["bb_eps"].append(two_bb2)
        req_inputs["speed_pos_bb"]["speed_eps"].append(three_speed1)
        req_inputs["speed_pos_bb"]["pos_eps"].append(three_pos2)
        req_inputs["speed_pos_bb"]["bb_eps"].append(three_bb3)
        req_inputs["speed_pos_price"]["speed_eps"].append(three_speed1)
        req_inputs["speed_pos_price"]["pos_eps"].append(three_pos2)
        req_inputs["speed_pos_price"]["price_eps"].append(three_price3)
        req_inputs["speed_bb_price"]["speed_eps"].append(three_speed1)
        req_inputs["speed_bb_price"]["bb_eps"].append(three_bb2)
        req_inputs["speed_bb_price"]["price_eps"].append(three_price3)
        req_inputs["pos_price_bb"]["pos_eps"].append(three_pos1)
        req_inputs["pos_price_bb"]["price_eps"].append(three_price2)
        req_inputs["pos_price_bb"]["bb_eps"].append(three_bb3)

    return req_inputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=str, required=True, help="sample size")    
    args = parser.parse_args()

    N = int(args.N)

    test_data = pd.read_pickle("esimft_data/pareto_problems.pkl")

    prep_data = prepare_req(test_data, N)

    with open('esimft_data/req_inputs_' + str(N) + '.pkl', 'wb') as f:
        pickle.dump(prep_data, f)
