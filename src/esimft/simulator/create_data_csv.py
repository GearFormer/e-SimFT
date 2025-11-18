import os
import csv
import pickle
import random
import json 

def json_reader(filename):
    with open(filename) as f:
        data=json.load(f)
    return data

def get_tokens():
    """
    This function reads the grammar.json files and returns two dictionary:
    1. tokens: which maps the tokens we have to unique indices
    2. inxs: which maps indices to tokens
    """
    inxs = {0:"<start>"}
    tokens = {"<start>":0}
    language = json_reader("language.json")
    language_vocabs = language['vocab']
    cnt = 1
    for i in language_vocabs:
        for j in language_vocabs[i]:
            if j not in tokens:
                tokens[j] = cnt
                inxs[cnt] = j
                cnt += 1
    tokens["<end>"] = cnt
    inxs[cnt] = "<end>"
    cnt += 1
    tokens["EOS"] = cnt
    inxs[cnt] = "EOS"
    return tokens, inxs


tokens, _ = get_tokens()

def name2inx(name):
    return tokens[name]
    
def load_pkl(file_name):
    with open(file_name,'rb') as pklfile:
        data = pickle.load(pklfile)
    return data

all_seq = 0
data_folder = "simulator_output"
all_files = os.listdir(data_folder)
for i in all_files:
    out = load_pkl(os.path.join(data_folder, i))
    all_seq += len(out)

print("all_data", all_seq)
val_numbers_to_select = int(0.05*0.01*all_seq)  # Ensure at least one number is selected
print("number of val samples: ", val_numbers_to_select)
val_to_select = random.sample(range(all_seq), val_numbers_to_select)
print(val_to_select)

test_numbers_to_select = int(0.05*0.01*all_seq)  # Ensure at least one number is selected
print("number of test samples: ", test_numbers_to_select)

test_to_select = []
while(len(test_to_select)< test_numbers_to_select):
    t_el = random.sample(range(all_seq), 1)
    t_el = t_el[0]
    if t_el not in val_to_select:
        if t_el not in test_to_select:
            test_to_select.append(t_el)

print(test_to_select)
csvfile_train = open("train_data.csv", 'w')
csvwriter_train = csv.writer(csvfile_train)

csvfile_val = open("val_data.csv", 'w')
csvwriter_val = csv.writer(csvfile_val)

csvfile_test = open("test_data.csv", 'w')
csvwriter_test = csv.writer(csvfile_test)

cnt = -1
cnt_v = -1
vv = 0
tt = 0
tr = 0
for i in all_files:
    cnt += 1
    if cnt%1000 == 0:
        print(cnt)

    out = load_pkl(os.path.join(data_folder, i))
    for j in out:
        cnt_v += 1
        motion_sign = 1
        target_length = len(out[j]["gear_train_sequence"])-1
        ratio = out[j]["output_motion_speed"]
        position = out[j]["output_position"]
        bb_min = out[j]["bounding_box_min"]
        bb_max = out[j]["bounding_box_max"]

        for k in range(3):
            if out[j]["output_motion_vector"][k] > 0.5 : 
                out[j]["output_motion_vector"][k] = 1
                xyz = k 
            elif out[j]["output_motion_vector"][k] < -0.5:
                out[j]["output_motion_vector"][k] = 1
                motion_sign = -1
                xyz = k 
            else: out[j]["output_motion_vector"][k] = 0
        
        weight_ = out[j]["weight"]
        price = out[j]["price"]

        r_t_0 = 0
        r_t_1 = 0

        if out[j]["input_motion_type"] == "T": r_t_0 = 1
        if out[j]["output_motion_type"] == "T": r_t_1 = 1

        x_train = [price, target_length, r_t_0, r_t_1, ratio, position[0], position[1], position[2], xyz, motion_sign,
                   bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2]]


        while (len(out[j]["gear_train_sequence"]) < 21):
            out[j]["gear_train_sequence"].append("EOS")
        x_train.append(list(map(name2inx, out[j]["gear_train_sequence"])))

        if cnt_v in val_to_select:
            vv += 1
            csvwriter_val.writerow(x_train)
        elif cnt_v in test_to_select:
            tt += 1
            csvwriter_test.writerow(x_train)
        else:
            csvwriter_train.writerow(x_train)
            tr += 1

print(vv, tt, tr)
