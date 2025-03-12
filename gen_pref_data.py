import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from train_models.utils.data_handle import load_data
import os
from train_models.utils.config_file import config
from train_models.load_model import loading_model
from train_models.utils.helper import is_grammatically_correct, is_physically_feasible
from simulator.gear_train_simulator import Simulator
from simulator.util.suppress_print import SuppressPrint
torch.set_printoptions(threshold=10_000)
import pandas as pd
from concurrent.futures import ThreadPoolExecutor



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GFModel:

    def __init__(self, input_size, args, max_length):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)))
        self.decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

    def run(self, input_batch):
        seq = ["<start>"]
        batch_size = len(input_batch)
        with torch.no_grad():
            input_ = torch.tensor(input_batch).to(torch.float32).to(device)
            encoded_input_ = self.encoder(input_)

            batch_prompt = torch.zeros((batch_size, len(seq))).long().to(device)
            for i in range(batch_size):
                for j in range(len(seq)):
                    batch_prompt[i, j] = self.get_dict.name2inx(seq[j])

            out_inx = self.decoder.generate(prompts=batch_prompt, context=encoded_input_, seq_len=21-len(seq), temperature=1.0)
                        
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

    # train_dataset = []
    # val_dataset = []

    gfm = GFModel(input_size, args, max_length)

    data = pd.read_pickle("esimft_data/pref_train.pkl")

    data_size = len(data.index)

    print("generating data...")
    dataset_price = []
    dataset_bb = []

    for i in range(0, data_size):

        print(i, " out of ", data_size)
        row = data.iloc[i]

        req_input_batch = []
        batch_size = 10
        for j in range(0, batch_size):
            req_input = []
            for k in range(2, 10):
                req_input.append(row.iloc[k])
            req_input_batch.append(req_input)

        seq_idx_batch, seq_batch = gfm.run(req_input_batch)

        sim_input_b = []
        for j in range(0, batch_size):
            sim_input_b.append({
                "id": j,
                "gear_train_sequence": seq_batch[j]
            })

        num_threads = 2
        # num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
            results = list(executor.map(run_simulator, sim_input_b))

        best_price_idx = -1
        best_bb_vol_idx = -1
        worst_price_idx = -1
        worst_bb_vol_idx = -1
        best_price = 1e9
        worst_price = 0
        best_bb_vol = 1e9
        worst_bb_vol = 0

        for i in range(0, len(results)):
            if results[i]["id"] == "failed":
                continue

            price = results[i]["price"]
            if  price < best_price:
                best_price = price
                best_price_idx = i
            if price > worst_price:
                worst_price = price
                worst_price_idx = i

            bb_vol = calculate_volume(results[i]["bounding_box_min"], results[i]["bounding_box_max"])
            if  bb_vol < best_bb_vol:
                best_bb_vol = bb_vol
                best_bb_vol_idx = i
            if bb_vol > worst_bb_vol:
                worst_bb_vol = bb_vol
                worst_bb_vol_idx = i
                
        mid_price = np.mean([best_price, worst_price])
        mid_bb_vol = np.mean([best_bb_vol, worst_bb_vol])

        # print(best_price, mid_price, worst_price)
        # print(best_bb_vol, mid_bb_vol, worst_bb_vol)
        # input()

        dataset_price.append((req_input, seq_idx_batch[best_price_idx], seq_idx_batch[worst_price_idx], mid_price))
        dataset_bb.append((req_input, seq_idx_batch[best_bb_vol_idx], seq_idx_batch[worst_bb_vol_idx], mid_bb_vol))

    print(data_size)
    print(len(dataset_price))
    print(len(dataset_bb))

    price_val_size = int(len(dataset_price) * 0.1)
    val_dataset_price = dataset_price[:price_val_size]
    train_dataset_price = dataset_price[price_val_size:]
    
    bb_val_size = int(len(dataset_bb) * 0.1)
    val_dataset_bb = dataset_bb[:bb_val_size]
    train_dataset_bb = dataset_bb[bb_val_size:]

    print()
    print("storing files...")

    df = pd.DataFrame(train_dataset_price, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle("esimft_data/pref_price_train.pkl")
    df = pd.DataFrame(val_dataset_price, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle("esimft_data/pref_price_val.pkl")

    df = pd.DataFrame(train_dataset_bb, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle("esimft_data/pref_bb_train.pkl")
    df = pd.DataFrame(val_dataset_bb, columns=['req_input', 'chosen_seq', 'reject_seq', 'mid_val'])
    df.to_pickle("esimft_data/pref_bb_val.pkl")