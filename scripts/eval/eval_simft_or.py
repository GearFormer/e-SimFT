import random
random.seed(0)
import numpy as np
np.random.seed(0)
import os
import torch
torch.manual_seed(0)
from train_models.utils.data_handle import load_data
from train_models.utils.config_file import config
from train_models.load_model import loading_model
from train_models.utils.helper import is_grammatically_correct, is_physically_feasible
from simulator.gear_train_simulator import Simulator
from esimft.utils.processing import SuppressPrint
torch.set_printoptions(threshold=10_000)
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import root_mean_squared_log_error
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GFModel:

    def __init__(self, input_size, args, max_length, encoder_path, decoder_path):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.encoder.load_state_dict(encoder_path)
        self.decoder.load_state_dict(decoder_path)
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

    def run(self, orig_req):
        seq = ["<start>"]
        batch_size = len(orig_req)
        with torch.no_grad():
            orig_req = torch.tensor(orig_req).to(torch.float32).to(device)
            orig_req = self.encoder(orig_req)
            
            encoded_input_ = orig_req

            batch_prompt = torch.zeros((batch_size, len(seq))).long().to(device)
            for i in range(batch_size):
                for j in range(len(seq)):
                    batch_prompt[i, j] = self.get_dict.name2inx(seq[j])

            out_inx = self.decoder.generate(prompts=batch_prompt, context=encoded_input_, seq_len=21-len(seq), temperature=0.0)
                        
            out_seq_batch = []
            out_inx_batch = []
            for i in range(batch_size):
                out_seq = ["<start>"] + list(map(get_dict.inx2name, out_inx[i].cpu().tolist())) + ['<end>']
                target_inx = out_seq.index("<end>")
                out_seq = out_seq[:target_inx+1]
                out_seq_batch.append(out_seq)

                out_inx_batch.append([0] + out_inx[i].long().tolist())

        return out_inx_batch, out_seq_batch 

def run_simulator(sim_input):

    simulator = Simulator()

    if not is_grammatically_correct(args, sim_input["gear_train_sequence"]):
        return {"id": "failed"}

    if not is_physically_feasible(sim_input["gear_train_sequence"], args.catalogue_path):
        return {"id": "failed"}

    try:
        results = simulator.run(sim_input)
    except:
        results = {"id": "failed"}
    
    return results


if __name__ == "__main__":
    args = config()
    simulator = Simulator()
    max_length = 21
    get_dict = load_data(args)
    input_size = 8

    test_data = pd.read_pickle("esimft_data/simft_test.pkl")
    data_size = len(test_data.index)
    
    orig_req_b = []
    target_objs = []

    encoder_path = torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name))
    decoder_path = torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name))

    gfm = GFModel(input_size, args, max_length, encoder_path, decoder_path)

    # print("gathering problems... ")
    for i in range(0, data_size):
        row = test_data.iloc[i]

        input_motion_type = row.iloc[2]
        output_motion_type = row.iloc[3]
        speed_ratio = row.iloc[4]
        output_position = np.array([row.iloc[5], row.iloc[6], row.iloc[7]])
        output_motion_direction = row.iloc[8]
        output_motion_sign = row.iloc[9]

        orig_req_b.append([input_motion_type, output_motion_type, speed_ratio, output_position[0], output_position[1], output_position[2],
                    output_motion_direction, output_motion_sign])
        
    # print("generating sequences... ")
    _, seq_pred_b = gfm.run(orig_req_b)

    # print("running simulation... ")
    sim_input_b = []
    for i in range(0, data_size):
        sim_input_b.append({
            "id": i,
            "gear_train_sequence": seq_pred_b[i]
        })

    num_threads = 32
    with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
        results = list(executor.map(run_simulator, sim_input_b))

    # print("computing metrics... ")
    speed_msle = 0
    pos_euc = 0
    valid = 0

    for i in range(0, data_size):
        if results[i]["id"] == "failed":
            continue
            
        if args.req_name == "speed":
            speed_msle += root_mean_squared_log_error([orig_req_b[i][2]], [results[i]["output_motion_speed"]])

        elif args.req_name == "pos":
            pos_euc += math.dist(results[i]["output_position"], [orig_req_b[i][3], orig_req_b[i][4], orig_req_b[i][5]])

        else:
            exit()

        valid += 1
    
    if args.req_name == "speed":
        print("average speed error", speed_msle/valid)

    elif args.req_name == "pos":
        print("average position error", pos_euc/valid)

    print("validity", valid/data_size)
