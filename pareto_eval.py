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
from simulator.util.suppress_print import SuppressPrint
torch.set_printoptions(threshold=10_000)
import pickle
from pymoo.indicators.hv import HV
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = config()
max_length = 21
get_dict = load_data(args)
input_size = 8

def crowding_distance(points):
    """
    Compute crowding distance for a set of Pareto-optimal points.
    
    :param points: (N, d) NumPy array where N is the number of points and d is the number of objectives.
    :return: Crowding distances for each point.
    """
    N, d = points.shape
    distances = np.zeros(N)

    for i in range(d):  # Iterate over each objective
        sorted_indices = np.argsort(points[:, i])
        sorted_points = points[sorted_indices, i]

        # Assign a large distance to boundary points
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Compute distances for middle points
        for j in range(1, N - 1):
            distances[sorted_indices[j]] += (sorted_points[j + 1] - sorted_points[j - 1])

    return distances

def spread_metric(pareto_points):
    pareto_points = np.array(pareto_points)
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]  # Sort by first objective
    
    distances = np.linalg.norm(np.diff(pareto_points, axis=0), axis=1)  # Euclidean distances
    d_f = np.linalg.norm(pareto_points[0] - pareto_points[-1])  # Distance between extremes
    mean_d = np.mean(distances)
    
    delta = (d_f + np.sum(np.abs(distances - mean_d))) / (d_f + (len(distances) * mean_d))
    return delta

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

def plot_objective_pairs(pairs, pareto_points, fname, title="Objective Value Pairs", xlabel="Objective 1", ylabel="Objective 2"):
    """
    Plots a list of objective value pairs on a scatter plot with optional title and axis labels.

    Parameters:
        pairs (list of tuple): List of objective value pairs [(x1, y1), (x2, y2), ...].
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Extract x and y values from pairs

    x_values = []
    po_x_values = []
    y_values = []
    po_y_values = []

    for pair in pairs:
        if pair in pareto_points:
            po_x_values.append(pair[0])
            po_y_values.append(pair[1])
        else:
            x_values.append(pair[0])
            y_values.append(pair[1])

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, color='blue', s=25)
    plt.scatter(po_x_values, po_y_values, color='red', s=25, label='Pareto optimal')

    # Adding labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax = plt.gca()
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 0.8])

    # Add grid and legend
    plt.grid(alpha=0.5)
    plt.legend()

    # Show the plot
    plt.savefig(fname)
    plt.close()

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

def eval_paretos(ref_point, results, scenario, method1, method2):

    print("Scenario: ", scenario, "\n")
    # print("Number of pareto points")
    # print(np.mean(results[scenario][method1]["no_pareto"]))
    # print(np.mean(results[scenario][method2]["no_pareto"]))
    # print(ttest_ind(results[scenario][method1]["no_pareto"], results[scenario][method2]["no_pareto"]))
    # print()
    
    hv_indicator = HV(ref_point=ref_point)
    hv1 = []
    hv2 = []
    spread1 = []
    spread2 = []
    no_p1 = []
    no_p2 = []

    for i in range(num_tests):
        hv1.append(hv_indicator(results[scenario][method1]["pareto"][i]))
        hv2.append(hv_indicator(results[scenario][method2]["pareto"][i]))

        no_p1.append(results[scenario][method1]["no_pareto"][i])
        no_p2.append(results[scenario][method2]["no_pareto"][i])

        spread1.append(spread_metric(results[scenario][method1]["pareto"][i]))
        spread2.append(spread_metric(results[scenario][method2]["pareto"][i]))

    print("Hypervolume")
    print(np.mean(hv1), np.std(hv1))
    print(np.mean(hv2), np.std(hv2))
    print(ttest_ind(hv1, hv2))
    print()

    print("No of points")
    print(np.mean(no_p1), np.std(no_p1))
    print(np.mean(no_p2), np.std(no_p2))
    print(ttest_ind(no_p1, no_p2))
    print()

    # print("Spread")
    # print(np.mean(spread1), np.std(spread1))
    # print(np.mean(spread2), np.std(spread2))
    # print(ttest_ind(spread1, spread2))
    # print()

def normalize(x, min, max):

    return (x - min) / (max - min)

def find_ref_points(num_tests, results):
    ref_points = {
        "speed": 0, 
        "pos": 0, 
        "price": 0, 
        "bb": 0
    }
    s = "speed_pos"
    for i in range(num_tests):
        pareto = results[s]["base"]["pareto"][i]
        for p in pareto:
            if p[0] > ref_points["speed"]:
                ref_points["speed"] = p[0]
            if p[1] > ref_points["pos"]:
                ref_points["pos"] = p[1]
    s = "price_bb"
    for i in range(num_tests):
        pareto = results[s]["base"]["pareto"][i]
        for p in pareto:
            if p[0] > ref_points["price"]:
                ref_points["price"] = p[0]
            if p[1] > ref_points["bb"]:
                ref_points["bb"] = p[1]

    return ref_points

def init_results(results, methods, scenarios, fname):
    for s in scenarios:
        for m in methods:
            results[s][m] = {
                "pareto": None,
                "no_pareto": None
            }
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    f.close()

def norm_results(m1, m2, scenarios, results):
    
    results_normalized = {}
    for s in scenarios:
        results_normalized[s] = {
            m1: {
                "no_pareto": None,
                "pareto": None
            },
            m2: {
                "no_pareto": None,
                "pareto": None
            }
        }
        if len(s.split("_")) == 2:
            (s1, s2) = s.split("_")
            pareto_norm_all1 = []
            pareto_norm_all2 = []
            for i in range(num_tests):
                pareto = results[s][m1]["pareto"][i]
                pareto_norm = []
                for p in pareto:
                    pareto_norm.append((normalize(p[0], 0, ref_points[s1]), normalize(p[1], 0, ref_points[s2])))
                pareto_norm = np.array(pareto_norm)
                pareto_norm_all1.append(pareto_norm)

                pareto = results[s][m2]["pareto"][i]
                pareto_norm = []
                for p in pareto:
                    pareto_norm.append((normalize(p[0], 0, ref_points[s1]), normalize(p[1], 0, ref_points[s2])))
                pareto_norm = np.array(pareto_norm)
                pareto_norm_all2.append(pareto_norm)

            results_normalized[s][m1]["no_pareto"] = results[s][m1]["no_pareto"]
            results_normalized[s][m1]["pareto"] = pareto_norm_all1
            results_normalized[s][m2]["no_pareto"] = results[s][m2]["no_pareto"]
            results_normalized[s][m2]["pareto"] = pareto_norm_all2

        elif len(s.split("_")) == 3:
            (s1, s2, s3) = s.split("_")
            pareto_norm_all1 = []
            pareto_norm_all2 = []
            for i in range(num_tests):
                pareto = results[s][m1]["pareto"][i]
                pareto_norm = []
                for p in pareto:
                    pareto_norm.append((normalize(p[0], 0, ref_points[s1]), normalize(p[1], 0, ref_points[s2]), normalize(p[2], 0, ref_points[s3])))
                pareto_norm = np.array(pareto_norm)
                pareto_norm_all1.append(pareto_norm)

                pareto = results[s][m2]["pareto"][i]
                pareto_norm = []
                for p in pareto:
                    pareto_norm.append((normalize(p[0], 0, ref_points[s1]), normalize(p[1], 0, ref_points[s2]), normalize(p[2], 0, ref_points[s3])))
                pareto_norm = np.array(pareto_norm)
                pareto_norm_all2.append(pareto_norm)

            results_normalized[s][m1]["no_pareto"] = results[s][m1]["no_pareto"]
            results_normalized[s][m1]["pareto"] = pareto_norm_all1
            results_normalized[s][m2]["no_pareto"] = results[s][m2]["no_pareto"]
            results_normalized[s][m2]["pareto"] = pareto_norm_all2

    return results_normalized

def eval_methods(m1, m2, scenarios, results, ref_points):

    results_normalized = norm_results(m1, m2, scenarios, results)

    for s in scenarios:
        if len(s.split("_")) == 2:
            (s1, s2) = s.split("_")
            ref_points = (1.0, 1.0)
            eval_paretos(ref_points, results_normalized, s, m1, m2)
        elif len(s.split("_")) == 3:
            (s1, s2, s3) = s.split("_")
            ref_points = (1.0, 1.0, 1.0)
            eval_paretos(ref_points, results_normalized, s, m1, m2)
        input()


def plot_2d_points(lists_of_points, filename, colors=None):
    """
    Plots four lists of 2D points using different colors and connects them with straight lines.

    Parameters:
    - lists_of_points: List of four lists, where each sublist contains tuples of (x, y) coordinates.
    - colors: List of four colors for the plots. If None, default colors are used.
    """

    methods = ["Baseline", "R. Soup", "R.-in-Context", "e-SimFT"]

    if colors is None:
        colors = ['red', 'orange', 'green', 'blue']
    
    plt.figure(figsize=(6, 6))
    
    for i, points in enumerate(lists_of_points):
        if len(points) == 0:
            continue  # Skip empty lists

        points = points[points[:, 0].argsort()]        
        x_values, y_values = zip(*points)  # Unpack points into x and y coordinates
        plt.scatter(x_values, y_values, color=colors[i], label=f'{methods[i]}', s=50)
        plt.plot(x_values, y_values, color=colors[i], linestyle='-', linewidth=1)
    
    (_, xlabel, ylabel, _) = filename.split("_")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Scatter Plot with Connected Lines')
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(filename)
    # plt.show()


def plot_3d_points(lists_of_points, filename, colors=None):
    """
    Plots four lists of 2D points using different colors and connects them with straight lines.

    Parameters:
    - lists_of_points: List of four lists, where each sublist contains tuples of (x, y) coordinates.
    - colors: List of four colors for the plots. If None, default colors are used.
    """

    methods = ["Baseline", "R. Soup", "R.-in-Context", "e-SimFT"]

    if colors is None:
        colors = ['red', 'orange', 'green', 'blue']
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, points in enumerate(lists_of_points):
        if len(points) == 0:
            continue  # Skip empty lists

        points = points[points[:, 0].argsort()]        
        x_values, y_values, z_values = points[:, 0], points[:, 1], points[:, 2]  # Extract x, y, z coordinates
        ax.scatter(x_values, y_values, z_values, color=colors[i], label=f'{methods[i]}', s=50)
        # ax.plot(x_values, y_values, z_values, color=colors[i], linestyle='-', linewidth=1)
    
        if len(points) >= 3:
            tri = Delaunay(points[:, :2])  # Delaunay triangulation for surface mesh
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=tri.simplices, color=colors[i], alpha=0.3)
    

    (_, xlabel, ylabel, zlable, _) = filename.split("_")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_zlabel(zlable, fontsize=16)
    # ax.set_xticks(fontsize=12)
    # ax.set_yticks(fontsize=12)
    # ax.set_zticks(fontsize=12)
    # plt.title('Scatter Plot with Connected Lines')
    ax.legend(fontsize=14)
    # ax.grid(True)
    plt.savefig(filename)
    # plt.show()


if __name__ == "__main__":

    num_tests = 30

    ref_points = {}
    ref_points["speed"] = 18690.32805080772
    ref_points["pos"] = 1.3493602375473808
    ref_points["price"] = 2374.112880506438
    ref_points["bb"] = 0.26230808838936703

    methods = ["base", "eps", "soup", "ric"]

    scenarios = ["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb",
                 "speed_pos_price", "speed_pos_bb", "speed_bb_price", "pos_price_bb"]

    with open("pareto_data_0.5-1.5_1.0-2.0.pkl", "rb") as f:
        results = pickle.load(f)
    f.close()

    # m1 = "soup"
    # m2 = "eps_sim"
    # eval_methods(m1, m2, scenarios, results, ref_points)


    res1 = norm_results("base", "soup", scenarios, results)
    res2 = norm_results("ric", "eps_sim", scenarios, results)

    s = scenarios[9]
    for t in range(30):
        points = []
        points.append(res1[s]["base"]["pareto"][t])
        points.append(res1[s]["soup"]["pareto"][t])
        points.append(res2[s]["ric"]["pareto"][t])
        points.append(res2[s]["eps_sim"]["pareto"][t])

        plot_3d_points(points, f"plot_position_cost_b.box_{t}.png")