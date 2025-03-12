import json
import numpy as np

def compute_weight(gear_train_seq):

    with open("/app/simulator/components/catalogue.json", 'r') as file:
        components = json.load(file)

    total_weight = 0
    for comp in gear_train_seq:
        if comp in components:
            total_weight += components[comp]["weight"]

    return total_weight

