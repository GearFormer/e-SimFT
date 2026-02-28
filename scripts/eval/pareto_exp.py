import numpy as np
import torch
import torch.nn as nn
from esimft.utils.config_file import config
from esimft.model.gearformer import GFModel, ObjEncoder, WeightEncoder
from esimft.model.gearformer_simft import GearFormerSimFT
from esimft.model.gearformer_soup import GearFormerSoup
from esimft.utils.gearformer.sim import run_simulator, calculate_volume
from esimft.utils.processing import SuppressPrint
from esimft.utils.data_handle import DataHandler
import pickle
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import os
from itertools import repeat


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

def prepare_inputs_and_prompts(orig_reqs, new_reqs_1=None, new_reqs_2=None, weights=None):

    inputs = ()

    orig_req_inputs = torch.tensor(orig_reqs, dtype=torch.float32, device=device)

    if new_reqs_1 is not None:
        new_req_inputs_1 = torch.tensor(new_reqs_1, dtype=torch.float32, device=device)
    else:
        new_req_inputs_1 = None

    if new_reqs_2 is not None:
        new_req_inputs_2 = torch.tensor(new_reqs_2, dtype=torch.float32, device=device)
    else:
        new_req_inputs_2 = None

    if weights is not None:
        weights_inputs = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        weights_inputs = None

    inputs = (orig_req_inputs, new_req_inputs_1, new_req_inputs_2, weights_inputs)

    start_token = 0
    prompts = torch.full(
        (orig_req_inputs.shape[0], 1),
        fill_value=start_token,
        dtype=torch.long,
        device=device,
    )

    return inputs, prompts
    
def translate_output(outputs, data_handler):
    solutions = []
    for pred in outputs:
        out_seq = ["<start>"] + list(map(data_handler.inx2name, pred.cpu().tolist())) + ['<end>']
        target_inx = out_seq.index("<end>")
        out_seq = out_seq[:target_inx+1]
        solutions.append(out_seq)
    return solutions

def eval_solutions(solutions):

    sim_inputs = []
    for i in range(len(solutions)):
        sim_inputs.append({
                "id": i,
                "gear_train_sequence": solutions[i]
        })
        
    with ThreadPoolExecutor(max_workers=config.num_threads_sim) as executor, SuppressPrint():
        results = list(executor.map(run_simulator, repeat(config), sim_inputs))

    return results

def find_obj_pairs(sim_results, req_input, scenario):

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

    obj_trips = []

    for i in range(0, min(len(sim_results), len(req_input))):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config()
    data_handler = DataHandler(config)

    ref_points = {}
    ref_points["speed"] = config.ref_pareto_speed
    ref_points["pos"] = config.ref_pareto_pos
    ref_points["price"] = config.ref_pareto_price
    ref_points["bb"] = config.ref_pareto_bb

    N = config.pareto_num_samples

    with open(os.path.join("esimft_data", config.data_pareto_samples_folder, f"req_inputs_{N}.pkl"), 'rb') as f:
        req_inputs = pickle.load(f)

    methods = config.test_methods
    scenarios = config.test_scenarios
    num_tests = config.pareto_exp_num_problems

    data_fname = os.path.join("esimft_data", config.data_pareto_samples_folder, f"pareto_data_{N}.pkl")
    results = {}
    for s in scenarios:
        results[s] = {}
        for m in methods:
            results[s][m] = {}

    gfm = GFModel(config, device, encoder_checkpoint_path=config.gearformer_encoder_checkpoint_name, 
                  decoder_checkpoint_path=config.gearformer_decoder_checkpoint_name)
    encoder = gfm.encoder
    decoder = gfm.decoder
    new_req_encoder = ObjEncoder(input_size=1, output_size=config.dim).to(device)
    weight_encoder = WeightEncoder(input_size=4, output_size=config.dim).to(device)
    
    for m in methods:
        print(f"Testing method {m}")

        if m == "base":
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

                    pred_seq = gfm.run(r_orig)[1] # [0] gives you the int
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

                    pred_seq1 = gfm.run(r1_orig)[1]
                    pred_seq2 = gfm.run(r2_orig)[1]
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

                if s == "speed_pos": 

                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        
                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)
                        
                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "price_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/2)]
                        r1_orig = r1[:,:8]
                        r1_new = r1[:,8:9]
                        r2 = req[int(N/2):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig, r1_new)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_bb": 
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    for i in range(0, num_tests):
                        print(i)
                        req = np.array(reqs[i])
                        r1 = req[:int(N/3)]
                        r1_orig = r1[:,:8]
                        r2 = req[int(N/3):]
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8:9]
                        r3 = req[int(N/3):]
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
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

                if s == "speed_pos": 
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["speed_pos"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_pos"]["pos_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_price":

                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["speed_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["speed_bb"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["pos_price"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["pos_bb"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "price_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["price_bb"]["price_eps"][:num_tests]
                    reqs2 = req_inputs["price_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]
                        r1_new = r1[:,8:9]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig, r1_new)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        obj_set = find_obj_pairs(sim_results1, r1, s) + find_obj_pairs(sim_results2, r2, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_bb": 
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

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
                        r3_new = r3[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_pos_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

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
                        r3_new = r3[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "speed_bb_price":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["speed_bb_price"]["speed_eps"][:num_tests]
                    reqs2 = req_inputs["speed_bb_price"]["bb_eps"][:num_tests]
                    reqs3 = req_inputs["speed_bb_price"]["price_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,9:10]

                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,8:9]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
                        sim_results3 = eval_solutions(pred_seq3)

                        obj_set = find_obj_triples(sim_results1, r1, s) + find_obj_triples(sim_results2, r2, s)
                        obj_set += find_obj_triples(sim_results3, r3, s)
                        pareto_points = np.array(pareto_frontier(np.array(obj_set)))

                        num_pareto.append(len(pareto_points))
                        pareto.append(pareto_points)

                elif s == "pos_price_bb":
                    model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                    model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

                    model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

                    model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                    model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

                    reqs1 = req_inputs["pos_price_bb"]["pos_eps"][:num_tests]
                    reqs2 = req_inputs["pos_price_bb"]["price_eps"][:num_tests]
                    reqs3 = req_inputs["pos_price_bb"]["bb_eps"][:num_tests]

                    for i in range(0, num_tests):
                        print(i)
                        r1 = np.array(reqs1[i])
                        r1_orig = r1[:,:8]

                        r2 = np.array(reqs2[i])
                        r2_orig = r2[:,:8]
                        r2_new = r2[:,8:9]

                        r3 = np.array(reqs3[i])
                        r3_orig = r3[:,:8]
                        r3_new = r3[:,9:10]

                        inputs1, prompts1 = prepare_inputs_and_prompts(r1_orig)
                        outputs1 = model_1.generate(inputs1, prompts1)
                        pred_seq1 = translate_output(outputs1, data_handler)
                        sim_results1 = eval_solutions(pred_seq1)

                        inputs2, prompts2 = prepare_inputs_and_prompts(r2_orig, r2_new)
                        outputs2 = model_2.generate(inputs2, prompts2)
                        pred_seq2 = translate_output(outputs2, data_handler)
                        sim_results2 = eval_solutions(pred_seq2)

                        inputs3, prompts3 = prepare_inputs_and_prompts(r3_orig, r3_new)
                        outputs3 = model_3.generate(inputs3, prompts3)
                        pred_seq3 = translate_output(outputs3, data_handler)
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

            two_w1 = config.two_reqs_weights_1
            two_w2 = config.two_reqs_weights_2
            two_n = int(N/6)

            three_w1 = config.three_reqs_weights_1
            three_w2 = config.three_reqs_weights_2
            three_w3 = config.three_reqs_weights_3

            three_n = np.linspace(0, N, 8, dtype=int).tolist()

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
                            soup_model_name = f"soup_{s1}_{two_w1[j]}_{s2}_{two_w2[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig)
                            outputs = soup_model.generate(inputs, prompts) 
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{two_w1[j]}_{s2}_{two_w2[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,8:9]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{two_w1[j]}_{s2}_{two_w2[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,9:10]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{two_w1[j]}_{s2}_{two_w2[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,8:9]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{two_w1[j]}_{s2}_{two_w2[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new = r[:,9:10]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{two_w1[j]}_{s2}_{two_w2[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[j*two_n:(j+1)*two_n]
                            r_orig = r[:,:8]
                            r_new_w1 = r[:,8:9]
                            r_new_w2 = r[:,9:10]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new_w1, r_new_w2)
                            outputs = soup_model.generate(inputs, prompts) 
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{three_w1[j]}_{s2}_{three_w2[j]}_{s3}_{three_w3[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_new = r[:,9:10]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{three_w1[j]}_{s2}_{three_w2[j]}_{s3}_{three_w3[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_new = r[:,8:9]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{three_w1[j]}_{s2}_{three_w2[j]}_{s3}_{three_w3[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_new_w1 = r[:,9:10]
                            r_new_w2 = r[:,8:9]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new_w1, r_new_w2)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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
                            soup_model_name = f"soup_{s1}_{three_w1[j]}_{s2}_{three_w2[j]}_{s3}_{three_w3[j]}.dict"
                            soup_model_path = os.path.join(config.checkpoint_path, "soup_models", soup_model_name)
                            soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                            soup_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, soup_model_path), map_location=device))

                            r = req[three_n[j]:three_n[j+1]]
                            r_orig = r[:,:8]
                            r_new_w1 = r[:,8:9]
                            r_new_w2 = r[:,9:10]

                            inputs, prompts = prepare_inputs_and_prompts(r_orig, r_new_w1, r_new_w2)
                            outputs = soup_model.generate(inputs, prompts)
                            pred_seq += translate_output(outputs, data_handler)

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

                ric_new_req_encoder = ObjEncoder(input_size=2, output_size=config.dim).to(device)
                ric_model = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ric_new_req_encoder, weight_encoder=weight_encoder, ric=True, device=device)
                ric_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, "ric.dict"), map_location=device))

                ric_weights_map = {
                    "speed_pos":       config.ric_speed_pos_weights,
                    "speed_price":     config.ric_speed_price_weights,
                    "speed_bb":        config.ric_speed_bb_weights,
                    "pos_price":       config.ric_pos_price_weights,
                    "pos_bb":          config.ric_pos_bb_weights,
                    "price_bb":        config.ric_price_bb_weights,
                    "speed_pos_bb":    config.ric_speed_pos_bb_weights,
                    "speed_pos_price": config.ric_speed_pos_price_weights,
                    "speed_bb_price":  config.ric_speed_bb_price_weights,
                    "pos_price_bb":    config.ric_pos_price_bb_weights,
                }
                w = np.array(ric_weights_map[s]).reshape(-1, 4).tolist()

                if num_scenarios == 2:
                    n = [int(x) for x in np.linspace(0, N, config.ric_2reqs_n_splits + 1)]
                else:
                    n = [int(x) for x in np.linspace(0, N, config.ric_3reqs_n_splits + 1)]

                for i in range(0, num_tests):
                    print(i)
                    req = np.array(reqs[i])
                    pred_seq = []

                    for j in range(len(w)):
                        r = req[n[j]:n[j+1]]
                        r_orig = r[:,:8]
                        r_new_1 = r[:,8:9]
                        r_new_2 = r[:,9:10]
                        weights = [w[j]] * (n[j+1]-n[j])

                        inputs, prompts = prepare_inputs_and_prompts(r_orig, new_reqs_1=r_new_1, new_reqs_2=r_new_2, weights=weights)
                        outputs = ric_model.generate(inputs, prompts) 
                        pred_seq += translate_output(outputs, data_handler)

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
