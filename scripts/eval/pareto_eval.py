import numpy as np
from train_models.utils.config_file import config
torch.set_printoptions(threshold=10_000)
import pickle
from pymoo.indicators.hv import HV
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


args = config()

def eval_paretos(ref_point, norm_results):
    
    hv_indicator = HV(ref_point=ref_point)
    hv = []

    for i in range(num_tests):
        hv.append(hv_indicator(norm_results[i]))

    print("Hypervolume")
    print(np.mean(hv), np.std(hv))
    print()

def ttest(ref_point, results, scenario, method1, method2):

    print("Scenario: ", scenario, "\n")
    
    hv_indicator = HV(ref_point=ref_point)
    hv1 = []
    hv2 = []

    for i in range(num_tests):
        hv1.append(hv_indicator(results[scenario][method1]["pareto"][i]))
        hv2.append(hv_indicator(results[scenario][method2]["pareto"][i]))

    print("T test results")
    print(ttest_ind(hv1, hv2))
    print()

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

def norm_results(scenario, method, results):
    
    if len(scenario.split("_")) == 2:
        (s1, s2) = scenario.split("_")
        pareto_norm_all = []
        for i in range(num_tests):
            pareto = results[scenario][method]["pareto"][i]
            pareto_norm = []
            for p in pareto:
                pareto_norm.append((normalize(p[0], 0, ref_points[s1]), normalize(p[1], 0, ref_points[s2])))
            pareto_norm = np.array(pareto_norm)
            pareto_norm_all.append(pareto_norm)

    elif len(scenario.split("_")) == 3:
        (s1, s2, s3) = s.split("_")
        pareto_norm_all = []
        for i in range(num_tests):
            pareto = results[scenario][method]["pareto"][i]
            pareto_norm = []
            for p in pareto:
                pareto_norm.append((normalize(p[0], 0, ref_points[s1]), normalize(p[1], 0, ref_points[s2]), normalize(p[2], 0, ref_points[s3])))
            pareto_norm = np.array(pareto_norm)
            pareto_norm_all.append(pareto_norm)

    return pareto_norm_all

def norm_results_pair(m1, m2, scenarios, results):
    
    results_normalized = {}
    for s in scenarios:
        results_normalized[s] = {
            m1: {
                "pareto": None
            },
            m2: {
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

            results_normalized[s][m1]["pareto"] = pareto_norm_all1
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

            results_normalized[s][m1]["pareto"] = pareto_norm_all1
            results_normalized[s][m2]["pareto"] = pareto_norm_all2

    return results_normalized

def eval_methods(scenario, method, results, ref_points):

    results_normalized = norm_results(scenario, method, results)

    if len(scenario.split("_")) == 2:
        ref_points = (1.0, 1.0)
    elif len(scenario.split("_")) == 3:
        ref_points = (1.0, 1.0, 1.0)
    eval_paretos(ref_points, results_normalized)
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

    methods = ["base", "soup", "ric", "sim", "eps", "eps_sim",]

    scenarios = ["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb",
                 "speed_pos_price", "speed_pos_bb", "speed_bb_price", "pos_price_bb"]

    with open(args.pareto_exp_data_path, "rb") as f:
        results = pickle.load(f)
    f.close()

    for s in scenarios:
        print("Scenario: ", s, "\n")
        for m in methods:
            print("Method: ", m, "\n")
            eval_methods(s, m, results, ref_points)

    # res1 = norm_results("base", "soup", scenarios, results)
    # res2 = norm_results("ric", "eps_sim", scenarios, results)
    # s = scenarios[9]
    # for t in range(30):
    #     points = []
    #     points.append(res1[s]["base"]["pareto"][t])
    #     points.append(res1[s]["soup"]["pareto"][t])
    #     points.append(res2[s]["ric"]["pareto"][t])
    #     points.append(res2[s]["eps_sim"]["pareto"][t])
    #     plot_3d_points(points, f"plot_position_cost_b.box_{t}.png")