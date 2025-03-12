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

    data1 = pd.read_pickle("esimft_data/sft_train.pkl")
    data2 = pd.read_pickle("esimft_data/pref_train.pkl")
    data = pd.concat([data1, data2], ignore_index=True)

    data_size = len(data.index)

    print("generating data...")
    dataset = []

    for i in range(0, data_size):

        print(i, " out of ", data_size)
        row = data.iloc[i]
        req_input = []
        for k in range(2, 10):
            req_input.append(row.iloc[k])
        target_speed = req_input[2]
        # print(row)
        # input()

        batch_size = 100
        # batch_size = 1
        req_input_batch = []
        for j in range(0, batch_size):
            req_input_batch.append(req_input)
            
        new_seq_idx_batch, new_seq_batch = gfm.run(req_input_batch)

        for j in range(0, batch_size):
            if not is_grammatically_correct(args, new_seq_batch[j]):
                continue
            if not is_physically_feasible(new_seq_batch[j], args.catalogue_path):
                continue

            input_data = {
                "gear_train_sequence": new_seq_batch[j],
                "id": 0
            }
            with SuppressPrint():
                new_speed = simulator.run(input_data)["output_motion_speed"]

            # dataset.append((req_input, new_seq_idx_batch[j]))
            # break

            if np.isclose(new_speed, target_speed, rtol=1e-6):
                dataset.append((req_input, new_seq_idx_batch[j]))
                break

    print(data_size)
    print(len(dataset))
    val_size = int(len(dataset) * 0.1)
    val_dataset = dataset[:val_size]
    train_dataset = dataset[val_size:]
    print()
    print("storing files...")

    df = pd.DataFrame(train_dataset, columns=['req_input', 'chosen_seq'])
    df.to_pickle("esimft_data/sft_speed_train.pkl")
    df = pd.DataFrame(val_dataset, columns=['req_input', 'chosen_seq'])
    df.to_pickle("esimft_data/sft_speed_val.pkl")
