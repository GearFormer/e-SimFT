import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
from train_models.utils.data_handle import load_data
from train_models.utils.config_file import config
from train_models.load_model import loading_model
from train_models.utils.helper import is_grammatically_correct, is_physically_feasible
from train_models.transformers import ObjEncoder, WeightEncoder
from simulator.gear_train_simulator import Simulator
from esimft.utils.processing import SuppressPrint
torch.set_printoptions(threshold=10_000)
import matplotlib.pyplot as plt
import pickle
from pymoo.indicators.hv import HV
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.stats import ttest_ind
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = config()
max_length = 21
get_dict = load_data(args)
input_size = 8


class GFModel:

    def __init__(self, encoder_name, decoder_name):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.encoder.load_state_dict(torch.load(encoder_name))
        self.decoder.load_state_dict(torch.load(decoder_name))
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()

    def run(self, req_input):
        with torch.no_grad():
            seq = ["<start>"]
            batch_size = len(req_input)
            with torch.no_grad():
                req_input = torch.tensor(req_input).to(torch.float32).to(device)
                encoded_input = self.encoder(req_input)

                batch_prompt = torch.zeros((batch_size, len(seq))).long().to(device)
                for i in range(batch_size):
                    for j in range(len(seq)):
                        batch_prompt[i, j] = self.get_dict.name2inx(seq[j])

                out_inx = self.decoder.generate(prompts=batch_prompt, context=encoded_input, seq_len=21-len(seq), temperature=1)
                            
                out_seq_batch = []
                # out_inx_batch = []
                for i in range(batch_size):
                    out_seq = ["<start>"] + list(map(get_dict.inx2name, out_inx[i].cpu().tolist())) + ['<end>']
                    target_inx = out_seq.index("<end>")
                    out_seq = out_seq[:target_inx+1]
                    if out_seq in out_seq_batch:
                        out_seq_batch.append(["<start>", "<end>"])
                    else:
                        out_seq_batch.append(out_seq)
                    # out_inx_batch.append([0] + out_inx[i].long().tolist())

        return out_seq_batch

class GFModel_obj:

    def __init__(self, encoder_name, decoder_name, obj_encoder_name):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.obj_encoder = ObjEncoder(input_size=1, output_size=args.dim)
        self.encoder.load_state_dict(torch.load(encoder_name))
        self.decoder.load_state_dict(torch.load(decoder_name))
        self.obj_encoder.load_state_dict(torch.load(obj_encoder_name))
        self.encoder.to(device)
        self.decoder.to(device)
        self.obj_encoder.to(device)
        self.encoder.eval()
        self.decoder.eval()
        self.obj_encoder.eval()

    def run(self, req_input, obj_input):
        with torch.no_grad():
            seq = ["<start>"]
            batch_size = len(req_input)
            with torch.no_grad():
                req_input = torch.tensor(req_input).to(torch.float32).to(device)
                obj_input = torch.tensor(obj_input).to(torch.float32).unsqueeze(-1).to(device)
                encoded_input = self.encoder(req_input) + self.obj_encoder(obj_input)

                batch_prompt = torch.zeros((batch_size, len(seq))).long().to(device)
                for i in range(batch_size):
                    for j in range(len(seq)):
                        batch_prompt[i, j] = self.get_dict.name2inx(seq[j])

                out_inx = self.decoder.generate(prompts=batch_prompt, context=encoded_input, seq_len=21-len(seq), temperature=1)
                            
                out_seq_batch = []
                # out_inx_batch = []
                for i in range(batch_size):
                    out_seq = ["<start>"] + list(map(get_dict.inx2name, out_inx[i].cpu().tolist())) + ['<end>']
                    target_inx = out_seq.index("<end>")
                    out_seq = out_seq[:target_inx+1]
                    if out_seq in out_seq_batch:
                        out_seq_batch.append(["<start>", "<end>"])
                    else:
                        out_seq_batch.append(out_seq)
                    # out_inx_batch.append([0] + out_inx[i].long().tolist())

        return out_seq_batch

class GFModel_obj2:

    def __init__(self, encoder_name, decoder_name, obj1_encoder_name, obj2_encoder_name):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.obj1_encoder = ObjEncoder(input_size=1, output_size=args.dim)
        self.obj2_encoder = ObjEncoder(input_size=1, output_size=args.dim)

        self.encoder.load_state_dict(torch.load(encoder_name))
        self.decoder.load_state_dict(torch.load(decoder_name))
        self.obj1_encoder.load_state_dict(torch.load(obj1_encoder_name))
        self.obj2_encoder.load_state_dict(torch.load(obj2_encoder_name))
        self.encoder.to(device)
        self.decoder.to(device)
        self.obj1_encoder.to(device)
        self.obj2_encoder.to(device)
        self.encoder.eval()
        self.decoder.eval()
        self.obj1_encoder.eval()
        self.obj2_encoder.eval()

    def run(self, req_input, obj1_input, obj2_input):
        with torch.no_grad():
            seq = ["<start>"]
            batch_size = len(req_input)
            with torch.no_grad():
                req_input = torch.tensor(req_input).to(torch.float32).to(device)
                obj1_input = torch.tensor(obj1_input).to(torch.float32).unsqueeze(-1).to(device)
                obj2_input = torch.tensor(obj2_input).to(torch.float32).unsqueeze(-1).to(device)

                encoded_input = self.encoder(req_input) + self.obj1_encoder(obj1_input) + self.obj2_encoder(obj2_input)

                batch_prompt = torch.zeros((batch_size, len(seq))).long().to(device)
                for i in range(batch_size):
                    for j in range(len(seq)):
                        batch_prompt[i, j] = self.get_dict.name2inx(seq[j])

                out_inx = self.decoder.generate(prompts=batch_prompt, context=encoded_input, seq_len=21-len(seq), temperature=1)
                            
                out_seq_batch = []
                # out_inx_batch = []
                for i in range(batch_size):
                    out_seq = ["<start>"] + list(map(get_dict.inx2name, out_inx[i].cpu().tolist())) + ['<end>']
                    target_inx = out_seq.index("<end>")
                    out_seq = out_seq[:target_inx+1]
                    if out_seq in out_seq_batch:
                        out_seq_batch.append(["<start>", "<end>"])
                    else:
                        out_seq_batch.append(out_seq)
                    # out_inx_batch.append([0] + out_inx[i].long().tolist())

        return out_seq_batch

class GFModel_w:

    def __init__(self, encoder_name, decoder_name, obj_encoder_name, w_encoder_name):
        self.get_dict = load_data(args)

        self.encoder, self.decoder = loading_model(args, input_size, self.get_dict.output_size, max_length)
        self.obj_encoder = ObjEncoder(input_size=2, output_size=args.dim)
        self.w_encoder = WeightEncoder(input_size=4, output_size=args.dim)

        self.encoder.load_state_dict(torch.load(encoder_name))
        self.decoder.load_state_dict(torch.load(decoder_name))
        self.obj_encoder.load_state_dict(torch.load(obj_encoder_name))
        self.w_encoder.load_state_dict(torch.load(w_encoder_name))

        self.encoder.to(device)
        self.decoder.to(device)
        self.obj_encoder.to(device)
        self.w_encoder.to(device)

        self.encoder.eval()
        self.decoder.eval()
        self.obj_encoder.eval()
        self.w_encoder.eval()

    def run(self, req_input, obj_input, w_input):
        with torch.no_grad():
            seq = ["<start>"]
            batch_size = len(req_input)
            with torch.no_grad():
                req_input = torch.tensor(req_input).to(torch.float32).to(device)
                obj_input = torch.tensor(obj_input).to(torch.float32).to(device)
                w_input = torch.tensor(w_input).to(torch.float32).to(device)
                encoded_input = self.encoder(req_input) + self.obj_encoder(obj_input) + self.w_encoder(w_input)

                batch_prompt = torch.zeros((batch_size, len(seq))).long().to(device)
                for i in range(batch_size):
                    for j in range(len(seq)):
                        batch_prompt[i, j] = self.get_dict.name2inx(seq[j])

                out_inx = self.decoder.generate(prompts=batch_prompt, context=encoded_input, seq_len=21-len(seq), temperature=1)
                            
                out_seq_batch = []
                # out_inx_batch = []
                for i in range(batch_size):
                    out_seq = ["<start>"] + list(map(get_dict.inx2name, out_inx[i].cpu().tolist())) + ['<end>']
                    target_inx = out_seq.index("<end>")
                    out_seq = out_seq[:target_inx+1]
                    if out_seq in out_seq_batch:
                        out_seq_batch.append(["<start>", "<end>"])
                    else:
                        out_seq_batch.append(out_seq)
                    # out_inx_batch.append([0] + out_inx[i].long().tolist())

        return out_seq_batch
    

def pareto_frontier(points):
    """
    Find the Pareto frontier (set of Pareto-optimal points).
    
    :param points: A NumPy array where each row is a point (obj1, obj2, obj3)
    :return: A set containing the Pareto-optimal points (as tuples)
    """
    pareto_set = set()
    num_points = points.shape[0]
    
    for i in range(num_points):
        dominated = False
        for j in range(num_points):
            if all(points[j] <= points[i]) and any(points[j] < points[i]):  # Strict domination check
                dominated = True
                break
        if not dominated:
            pareto_set.add(tuple(points[i]))  # Convert to tuple for set operations
    
    return list(pareto_set)


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

def run_simulator(sim_input):

    simulator = Simulator()

    if len(sim_input["gear_train_sequence"]) < 4:
        return {"id": "failed"}

    if not is_grammatically_correct(args, sim_input["gear_train_sequence"]):
        return {"id": "failed"}

    if not is_physically_feasible(sim_input["gear_train_sequence"], args.catalogue_path):
        return {"id": "failed"}

    try:
        results = simulator.run(sim_input)
    except:
        results = {"id": "failed"}
    
    return results

def eval_solutions(solutions):

    sim_inputs = []
    for i in range(len(solutions)):
        sim_inputs.append({
                "id": i,
                "gear_train_sequence": solutions[i]
        })

    num_threads = 32
    with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
        results = list(executor.map(run_simulator, sim_inputs))

    return results

def find_obj_pairs(sim_results, req_input, scenario):

    # scenarios = ["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb",
    #              "speed_pos_bb", "speed_pos_price", "speed_bb_price", "pos_price_bb"]
    
    obj_pairs = []

    for i in range(0, len(sim_results)):
        if sim_results[i]["id"] == "failed":
            continue

        r = req_input[i]
        
        actual_speed = sim_results[i]["output_motion_speed"]
        target_speed = r[2]

        actual_pos = sim_results[i]["output_position"]
        target_pos = np.array([r[3], r[4], r[5]])

        actual_price = sim_results[i]["price"]
        target_price = r[8]
        
        actual_bb = calculate_volume(sim_results[i]["bounding_box_min"], sim_results[i]["bounding_box_max"])
        target_bb = r[9]
        
        if scenario == "speed_pos": 
            diff1 = np.sqrt((target_speed - actual_speed) ** 2)
            diff2 = np.linalg.norm(target_pos - actual_pos)

        elif scenario == "speed_price":             
            diff1 = np.sqrt((target_speed - actual_speed) ** 2)
            diff2 = np.sqrt((target_price - actual_price) ** 2)

        elif scenario == "speed_bb":
            diff1 = np.sqrt((target_speed - actual_speed) ** 2)
            diff2 = np.sqrt((target_bb - actual_bb) ** 2)

        elif scenario == "pos_price":
            diff1 = np.linalg.norm(target_pos - actual_pos)
            diff2 = np.sqrt((target_price - actual_price) ** 2)

        elif scenario == "pos_bb":
            diff1 = np.linalg.norm(target_pos - actual_pos)
            diff2 = np.sqrt((target_bb - actual_bb) ** 2)

        elif scenario == "price_bb":
            diff1 = np.sqrt((target_price - actual_price) ** 2)
            diff2 = np.sqrt((target_bb - actual_bb) ** 2)

        obj_pairs.append((diff1, diff2))

    return obj_pairs

def find_obj_triples(sim_results, req_input, scenario):

    # scenarios = ["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb",
    #              "speed_pos_bb", "speed_pos_price", "speed_bb_price", "pos_price_bb"]
    
    obj_trips = []

    for i in range(0, len(req_input)):
        if sim_results[i]["id"] == "failed":
            continue

        r = req_input[i]
        
        actual_speed = sim_results[i]["output_motion_speed"]
        target_speed = r[2]

        actual_pos = sim_results[i]["output_position"]
        target_pos = np.array([r[3], r[4], r[5]])

        actual_price = sim_results[i]["price"]
        target_price = r[8]
        
        actual_bb = calculate_volume(sim_results[i]["bounding_box_min"], sim_results[i]["bounding_box_max"])
        target_bb = r[9]
        
        if scenario == "speed_pos_bb": 
            diff1 = np.sqrt((target_speed - actual_speed) ** 2)
            diff2 = np.linalg.norm(target_pos - actual_pos)
            diff3 = np.sqrt((target_bb - actual_bb) ** 2)

        elif scenario == "speed_pos_price":             
            diff1 = np.sqrt((target_speed - actual_speed) ** 2)
            diff2 = np.linalg.norm(target_pos - actual_pos)
            diff3 = np.sqrt((target_price - actual_price) ** 2)

        elif scenario == "speed_bb_price":
            diff1 = np.sqrt((target_speed - actual_speed) ** 2)
            diff2 = np.sqrt((target_bb - actual_bb) ** 2)
            diff3 = np.sqrt((target_price - actual_price) ** 2)

        elif scenario == "pos_price_bb":
            diff1 = np.linalg.norm(target_pos - actual_pos)
            diff2 = np.sqrt((target_price - actual_price) ** 2)
            diff3 = np.sqrt((target_bb - actual_bb) ** 2)

        obj_trips.append((diff1, diff2, diff3))

    return obj_trips


if __name__ == "__main__":


    ref_points = {}
    ref_points["speed"] = 18690.32805080772
    ref_points["pos"] = 1.3493602375473808
    ref_points["price"] = 2374.112880506438
    ref_points["bb"] = 0.26230808838936703

    N = args.N

    with open('esimft_data/req_inputs_' + str(N) + '.pkl', 'rb') as f:
        req_inputs = pickle.load(f)

    methods = ["base", "sim", "eps", "eps_sim", "soup", "ric"]
    scenarios = ["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb",
                 "speed_pos_price", "speed_pos_bb", "speed_bb_price", "pos_price_bb"]

    num_tests = 30

    data_fname = "esimft_data/pareto_data_" + str(N) + ".pkl"
    results = {}
    for s in scenarios:
        results[s] = {}
        for m in methods:
            results[s][m] = {}
    # with open(data_fname, "rb") as f:
    #     results = pickle.load(f)
    # f.close()

    for m in methods:
        print(m)

        if m == "base":
            encoder_path = "/app/train_models/models/GearFormer_0.0001_18_encoder.dict"
            decoder_path = "/app/train_models/models/GearFormer_0.0001_18_decoder.dict"
            gfm = GFModel(encoder_path, decoder_path)

            for s in scenarios:
                t0 = time.time()
                print(s)
                num_scenarios = len(s.split("_"))
                if num_scenarios == 2:
                    reqs = req_inputs["default2"][:num_tests]
                else:
                    reqs = req_inputs["default3"][:num_tests]

                num_pareto = []
                pareto = []
                for i in range(0, num_tests):
                    print(i)
                    r = np.array(reqs[i])
                    r_orig = r[:,:8]

                    pred_seq = gfm.run(r_orig)
                    sim_results = eval_solutions(pred_seq)
                    if num_scenarios == 2:
                        obj_set = find_obj_pairs(sim_results, r, s)
                    elif num_scenarios == 3:
                        obj_set = find_obj_triples(sim_results, r, s)
                    pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                    num_pareto.append(len(pareto_points))
                    pareto.append(pareto_points)

                print(datetime.now().strftime("%H:%M:%S"))
                print()

                results[s][m]["pareto"] = pareto
                with open(data_fname, "wb") as f:
                    pickle.dump(results, f)
                f.close()

        elif m == "eps":
            encoder_path = "/app/train_models/models/GearFormer_0.0001_18_encoder.dict"
            decoder_path = "/app/train_models/models/GearFormer_0.0001_18_decoder.dict"
            gfm = GFModel(encoder_path, decoder_path)

            for s in scenarios:
                t0 = time.time()
                print(s)
                num_scenarios = len(s.split("_"))

                num_pareto = []
                pareto = []

                if s == "speed_pos": 
                    reqs1 = req_inputs["speed_pos"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos"]["pos_eps"][:num_tests]

                elif s == "speed_price":
                    reqs1 = req_inputs["speed_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_price"]["price_eps"][:num_tests]

                elif s == "speed_bb":
                    reqs1 = req_inputs["speed_bb"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_bb"]["bb_eps"][:num_tests]

                elif s == "pos_price":
                    reqs1 = req_inputs["pos_price"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_price"]["price_eps"][:num_tests]

                elif s == "pos_bb":
                    reqs1 = req_inputs["pos_bb"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_bb"]["bb_eps"][:num_tests]

                elif s == "price_bb":
                    reqs1 = req_inputs["price_bb"]["price_eps"][:num_tests]
                    reqs2 = req_inputs["price_bb"]["bb_eps"][:num_tests]

                elif s == "speed_pos_bb": 
                    reqs1 = req_inputs["speed_pos_bb"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos_bb"]["pos_eps"][:num_tests]
                    reqs3 = req_inputs["speed_pos_bb"]["bb_eps"][:num_tests]

                elif s == "speed_pos_price":
                    reqs1 = req_inputs["speed_pos_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos_price"]["pos_eps"][:num_tests]
                    reqs3 = req_inputs["speed_pos_price"]["price_eps"][:num_tests]

                elif s == "speed_bb_price":
                    reqs1 = req_inputs["speed_bb_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_bb_price"]["bb_eps"][:num_tests]
                    reqs3 = req_inputs["speed_bb_price"]["price_eps"][:num_tests]

                elif s == "pos_price_bb":
                    reqs1 = req_inputs["pos_price_bb"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_price_bb"]["price_eps"][:num_tests]
                    reqs3 = req_inputs["pos_price_bb"]["bb_eps"][:num_tests]

                for i in range(0, num_tests):
                    print(i)
                    r1 = np.array(reqs1[i])
                    r1_orig = r1[:,:8]
                    r2 = np.array(reqs2[i])
                    r2_orig = r2[:,:8]

                    pred_seq1 = gfm.run(r1_orig)
                    pred_seq2 = gfm.run(r2_orig)
                    sim_results1 = eval_solutions(pred_seq1)
                    sim_results2 = eval_solutions(pred_seq2)
                    
                    if num_scenarios == 2:
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)

                    elif num_scenarios == 3:
                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]

                        pred_seq3 = gfm.run(r3_orig)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)

                    pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                    num_pareto.append(len(pareto_points))
                    pareto.append(pareto_points)

                print(datetime.now().strftime("%H:%M:%S"))
                print()

                results[s][m]["pareto"] = pareto
                with open(data_fname, "wb") as f:
                    pickle.dump(results, f)
                f.close()

        elif m == "sim":
            for s in scenarios:
                t0 = time.time()
                print(s)
                num_scenarios = len(s.split("_"))

                if num_scenarios == 2:
                    reqs = req_inputs["default2"][:num_tests]
                else:
                    reqs = req_inputs["default3"][:num_tests]

                num_pareto = []
                pareto = []

                encoder_path = "/app/train_models/models/GearFormer_0.0001_18_encoder.dict"

                if s == "speed_pos": 
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm2 = GFModel(encoder_path, decoder2_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_price":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price":
                    decoder1_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_bb":
                    decoder1_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "price_bb":
                    decoder1_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    obj_encoder_path1 = "/app/train_models/models/SFT_price_new_encoder.dict"
                    gfm1 = GFModel_obj(encoder_path, decoder1_path, obj_encoder_path1)
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    obj_encoder_path2 = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path2)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r1_new = r2[:,8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        pred_seq1 = gfm1.run(r1_orig, r1_new)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_bb": 
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm2 = GFModel(encoder_path, decoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_price":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm2 = GFModel(encoder_path, decoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb_price":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    obj_encoder2_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price_bb":
                    decoder1_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    obj_encoder2_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    decoder2_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                print(datetime.now().strftime("%H:%M:%S"))
                print()

                results[s][m]["pareto"] = pareto
                with open(data_fname, "wb") as f:
                    pickle.dump(results, f)
                f.close()

        elif m == "eps_sim":

            for s in scenarios:
                t0 = time.time()
                print(s)

                num_pareto = []
                pareto = []

                encoder_path = "/app/train_models/models/GearFormer_0.0001_18_encoder.dict"

                if s == "speed_pos": 
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm2 = GFModel(encoder_path, decoder2_path)

                    reqs1 = req_inputs["speed_pos"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos"]["pos_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_price":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    reqs1 = req_inputs["speed_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    reqs1 = req_inputs["speed_bb"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price":
                    decoder1_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    reqs1 = req_inputs["pos_price"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_bb":
                    decoder1_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    obj_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path)

                    reqs1 = req_inputs["pos_bb"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "price_bb":
                    decoder1_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    obj_encoder_path1 = "/app/train_models/models/SFT_price_new_encoder.dict"
                    gfm1 = GFModel_obj(encoder_path, decoder1_path, obj_encoder_path1)
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    obj_encoder_path2 = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder_path2)

                    reqs1 = req_inputs["price_bb"]["price_eps"][:num_tests]
                    reqs2 = req_inputs["price_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]
                        r1_new = r1[:,8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        pred_seq1 = gfm1.run(r1_orig, r1_new)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_bb": 
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm2 = GFModel(encoder_path, decoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    reqs1 = req_inputs["speed_pos_bb"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos_bb"]["pos_eps"][:num_tests]
                    reqs3 = req_inputs["speed_pos_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]

                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_price":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    decoder2_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm2 = GFModel(encoder_path, decoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    reqs1 = req_inputs["speed_pos_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos_price"]["pos_eps"][:num_tests]
                    reqs3 = req_inputs["speed_pos_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]

                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb_price":
                    decoder1_path = "/app/train_models/models/SFT_speed_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    obj_encoder2_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    decoder2_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    reqs1 = req_inputs["speed_bb_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_bb_price"]["bb_eps"][:num_tests]
                    reqs3 = req_inputs["speed_bb_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9]

                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,8]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price_bb":
                    decoder1_path = "/app/train_models/models/SFT_pos_decoder.dict"
                    gfm1 = GFModel(encoder_path, decoder1_path)
                    obj_encoder2_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    decoder2_path = "/app/train_models/models/DPO_price_12_decoder.dict"
                    gfm2 = GFModel_obj(encoder_path, decoder2_path, obj_encoder2_path)
                    obj_encoder3_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                    decoder3_path = "/app/train_models/models/DPO_bb_11_decoder.dict"
                    gfm3 = GFModel_obj(encoder_path, decoder3_path, obj_encoder3_path)

                    reqs1 = req_inputs["pos_price_bb"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_price_bb"]["price_eps"][:num_tests]
                    reqs3 = req_inputs["pos_price_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8]

                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9]

                        pred_seq1 = gfm1.run(r1_orig)
                        pred_seq2 = gfm2.run(r2_orig, r2_new)
                        pred_seq3 = gfm3.run(r3_orig, r3_new)
                        sim_results1 = eval_solutions(pred_seq1)
                        sim_results2 = eval_solutions(pred_seq2)
                        sim_results3 = eval_solutions(pred_seq3)
                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                print(datetime.now().strftime("%H:%M:%S"))
                print()

                results[s][m]["pareto"] = pareto
                with open(data_fname, "wb") as f:
                    pickle.dump(results, f)
                f.close()

        elif m == "soup":

            two_w1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            two_w2 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
            two_n = int(N/6)

            three_w1 = [0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.33]
            three_w2 = [0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.33]
            three_w3 = [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.33]

            if args.N == 30:
                three_n = [0, 4, 8, 12, 16, 20, 24, 30]
            elif args.N == 300:
                three_n = [0, 40, 80, 120, 160, 200, 240, 300]
            else:
                three_n = np.linspace(0, N, 8).tolist()

            for s in scenarios:
                t0 = time.time()
                print(s)

                num_scenarios = len(s.split("_"))

                if num_scenarios == 2:
                    reqs = req_inputs["default2"][:num_tests]
                else:
                    reqs = req_inputs["default3"][:num_tests]

                hv = []
                num_pareto = []
                pareto = []

                encoder_path = "/app/train_models/models/GearFormer_0.0001_18_encoder.dict"
                path_default = "/app/train_models/models/"

                if num_scenarios == 2:
                    (s1, s2) = s.split("_")
                else:
                    (s1, s2, s3) = s.split("_")

                if s == "speed_pos":
                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(two_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(two_w1[j]) + "_" + s2 + "_" + str(two_w2[j]) + "_decoder.dict"
                            gfm = GFModel(encoder_path, decoder_path)
                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            pred_seq += gfm.run(r_orig)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_pairs(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_price":
                    obj_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(two_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(two_w1[j]) + "_" + s2 + "_" + str(two_w2[j]) + "_decoder.dict"
                            gfm = GFModel_obj(encoder_path, decoder_path, obj_encoder_path)
                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,8]
                            pred_seq += gfm.run(r_orig, r_new)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_pairs(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)
                    
                elif s == "speed_bb":
                    obj_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(two_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(two_w1[j]) + "_" + s2 + "_" + str(two_w2[j]) + "_decoder.dict"
                            gfm = GFModel_obj(encoder_path, decoder_path, obj_encoder_path)
                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,9]
                            pred_seq += gfm.run(r_orig, r_new)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_pairs(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)
                    
                elif s == "pos_price":
                    obj_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(two_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(two_w1[j]) + "_" + s2 + "_" + str(two_w2[j]) + "_decoder.dict"
                            gfm = GFModel_obj(encoder_path, decoder_path, obj_encoder_path)
                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,8]
                            pred_seq += gfm.run(r_orig, r_new)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_pairs(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_bb":
                    obj_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(two_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(two_w1[j]) + "_" + s2 + "_" + str(two_w2[j]) + "_decoder.dict"
                            gfm = GFModel_obj(encoder_path, decoder_path, obj_encoder_path)
                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,9]
                            pred_seq += gfm.run(r_orig, r_new)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_pairs(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "price_bb":
                    obj1_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                    obj2_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(two_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(two_w1[j]) + "_" + s2 + "_" + str(two_w2[j]) + "_decoder.dict"
                            gfm = GFModel_obj2(encoder_path, decoder_path, obj1_encoder_path, obj2_encoder_path)
                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_netwo_w1 = r[:,8]
                            r_netwo_w2 = r[:,9]
                            pred_seq += gfm.run(r_orig, r_netwo_w1, r_netwo_w2)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_pairs(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_bb":
                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(three_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(three_w1[j]) + "_" + s2 + "_" + str(three_w2[j]) + "_" + s3 + "_" + str(three_w3[j]) + "_decoder.dict"
                            obj3_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                            gfm = GFModel_obj(encoder_path, decoder_path, obj3_encoder_path)
                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_new = req[:,9]
                            pred_seq += gfm.run(r_orig, r_new)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_triples(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_price":
                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(three_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(three_w1[j]) + "_" + s2 + "_" + str(three_w2[j]) + "_" + s3 + "_" + str(three_w3[j]) + "_decoder.dict"
                            obj3_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                            gfm = GFModel_obj(encoder_path, decoder_path, obj3_encoder_path)
                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_new = req[:,8]
                            pred_seq += gfm.run(r_orig, r_new)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_triples(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb_price":
                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(three_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(three_w1[j]) + "_" + s2 + "_" + str(three_w2[j]) + "_" + s3 + "_" + str(three_w3[j]) + "_decoder.dict"
                            obj2_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                            obj3_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                            gfm = GFModel_obj2(encoder_path, decoder_path, obj2_encoder_path, obj3_encoder_path)
                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_netwo_w1 = req[:,9]
                            r_netwo_w2 = req[:,8]
                            pred_seq += gfm.run(r_orig, r_netwo_w1, r_netwo_w2)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_triples(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price_bb":
                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        pred_seq = []

                        for j in range(len(three_w1)):
                            decoder_path = path_default + "soup_" + s1 + "_" + str(three_w1[j]) + "_" + s2 + "_" + str(three_w2[j]) + "_" + s3 + "_" + str(three_w3[j]) + "_decoder.dict"
                            obj2_encoder_path = "/app/train_models/models/SFT_price_new_encoder.dict"
                            obj3_encoder_path = "/app/train_models/models/SFT_bb_new_encoder.dict"
                            gfm = GFModel_obj2(encoder_path, decoder_path, obj2_encoder_path, obj3_encoder_path)
                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_netwo_w1 = req[:,8]
                            r_netwo_w2 = req[:,9]
                            pred_seq += gfm.run(r_orig, r_netwo_w1, r_netwo_w2)

                        sim_results = eval_solutions(pred_seq)
                        obj_set = find_obj_triples(sim_results, req, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                print(datetime.now().strftime("%H:%M:%S"))
                print()

                results[s][m]["pareto"] = pareto
                with open(data_fname, "wb") as f:
                    pickle.dump(results, f)
                f.close()

        elif m == "ric":

            for s in scenarios:
                t0 = time.time()
                print(s)

                num_scenarios = len(s.split("_"))

                if num_scenarios == 2:
                    reqs = req_inputs["default2"][:num_tests]
                else:
                    reqs = req_inputs["default3"][:num_tests]

                num_pareto = []
                pareto = []

                encoder_path = "/app/train_models/models/GearFormer_0.0001_18_encoder.dict"
                obj_encoder_path = "/app/train_models/models/SFT_ric_obj_encoder.dict"
                w_encoder_path = "/app/train_models/models/SFT_ric_w_encoder.dict"
                decoder_path = "/app/train_models/models/SFT_ric_decoder.dict"

                gfm = GFModel_w(encoder_path, decoder_path, obj_encoder_path, w_encoder_path)

                if s == "speed_pos":
                    w = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0]]
                elif s == "speed_price":
                    w = [[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]]
                elif s == "speed_bb":
                    w = [[1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 1]]
                elif s == "pos_price":
                    w = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0]]
                elif s == "pos_bb":
                    w = [[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]]
                elif s == "price_bb":
                    w = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]]
                if s == "speed_pos_bb":
                    w = [[1, 0, 0 ,0], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 1]]
                elif s == "speed_pos_price":
                    w = [[1, 0, 0 ,0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 0]]
                elif s == "speed_bb_price":
                    w = [[1, 0, 0 ,0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]]
                elif s == "pos_price_bb":
                    w = [[0, 1, 0 ,0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]


            
                if num_scenarios == 2:
                    if args.N == 30:
                        n = [0, 10, 20, 30]
                    elif args.N == 300:
                        n = [0, 100, 200, 300]
                    else:
                        n = np.linspace(0, N, 4).tolist()
                else:
                    if args.N == 30:
                        n = [0, 4, 8, 12, 16, 20, 24, 30]
                    elif args.N == 300:
                        n = [0, 40, 80, 120, 160, 200, 240, 300]
                    else:
                        n = np.linspace(0, N, 8).tolist()

                for i in range(0, num_tests):
                    print(i)
                    req = np.array(reqs[i])
                    pred_seq = []

                    for j in range(len(w)):
                        gfm = GFModel_w(encoder_path, decoder_path, obj_encoder_path, w_encoder_path)
                        r = req[n[j]:n[j+1]]
                        r_orig = r[:,:8]
                        r_new = r[:,8:10]
                        pred_seq += gfm.run(r_orig, r_new, [w[j]] * (n[j+1]-n[j]))
                   
                    sim_results = eval_solutions(pred_seq)
                    if num_scenarios == 2:
                        obj_set = find_obj_pairs(sim_results, req, s)
                    else:
                        obj_set = find_obj_triples(sim_results, req, s)
                    pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                    num_pareto.append(len(pareto_points))
                    pareto.append(pareto_points)

                print(datetime.now().strftime("%H:%M:%S"))
                print()

                results[s][m]["pareto"] = pareto
                with open(data_fname, "wb") as f:
                    pickle.dump(results, f)
                f.close()
    
        else:
            continue
