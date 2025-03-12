import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
from train_models.utils.data_handle import load_data
import os
from train_models.utils.config_file_eval_simft_obj import config
from train_models.load_model import loading_model
from train_models.utils.helper import is_grammatically_correct, is_physically_feasible
from simulator.gear_train_simulator import Simulator
from simulator.util.suppress_print import SuppressPrint
torch.set_printoptions(threshold=10_000)
from concurrent.futures import ThreadPoolExecutor
from models.transformers import ObjEncoder

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
    

class GFModel_bb:

    def __init__(self, input_size, args, max_length, encoder_path, decoder_path, encoder_obj_path):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.encoder.load_state_dict(encoder_path)
        self.decoder.load_state_dict(decoder_path)
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

        self.encoder_obj = ObjEncoder(input_size=1, output_size=args.dim)
        self.encoder_obj.load_state_dict(encoder_obj_path)
        self.encoder_obj.to(device)
        self.encoder_obj.eval()

    def run(self, orig_req, obj_req):
        seq = ["<start>"]
        batch_size = len(orig_req)
        with torch.no_grad():
            orig_req = torch.tensor(orig_req).to(torch.float32).to(device)
            orig_req = self.encoder(orig_req)
            
            # encoded_input_ = orig_req

            obj_req = torch.tensor(obj_req).to(torch.float32).to(device)
            obj_req = self.encoder_obj(obj_req)

            encoded_input_ = orig_req + obj_req

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

    encoder_path = torch.load("/app/gearformer_model/models/Xtransformer_0.0001_18_encoder.dict")
    decoder_path = torch.load("/app/gearformer_model/models/Xtransformer_0.0001_18_decoder.dict")
    decoder_bb_path = torch.load("/app/gearformer_model/models/PPO_bb_4_decoder.dict")
    encoder_obj_path = torch.load("/app/gearformer_model/models/SFT_bb_encoder.dict")

    gfm_bb = GFModel_bb(input_size, args, max_length, encoder_path, decoder_bb_path, encoder_obj_path)
    gfm = GFModel(input_size, args, max_length, encoder_path, decoder_path)

    """
    input_[0]: input_ motion type, 1 for T and 0 for R
    input_[1]: output motion type, 1 for T and 0 for R
    input_[2]: speed ratio
    input_[3], input_[4], input_[5]: x, y, z for output position
    input_[6]: output motion vector direction xyz - 0 for x, 1 for y and 2 for z
    input_[7] : output motion vector sign 
    """

    req_input = [[0, 0, 1.35, 0.1, 0.15, 0.2, 0, 1]]
    bb_input = [[0.08]]

    # print("generating sequences... ")
    _, seq = gfm.run(req_input)
    _, seq_bb = gfm_bb.run(req_input, bb_input)

    print(seq)
    print(seq_bb)

    sim_input_b = []
    sim_input_b.append({
        "id": 0,
        "gear_train_sequence": seq[0]
    })
    sim_input_b.append({
        "id": 1,
        "gear_train_sequence": seq_bb[0]
    })

    num_threads = 2
    with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
        results = list(executor.map(run_simulator, sim_input_b))

    print(calculate_volume(results[0]["bounding_box_min"], results[0]["bounding_box_max"]))
    print(calculate_volume(results[1]["bounding_box_min"], results[1]["bounding_box_max"]))
