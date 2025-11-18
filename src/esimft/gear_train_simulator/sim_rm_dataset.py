from esimft.gear_train_simulator.simulator import Simulator
from util.processing import SuppressPrint
import h5py
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
import h5py
import time
import json
from utils.data_handle import load_data
from utils.config_file import config

simulator = Simulator()

def json_reader(filename):
    with open(filename) as f:
        data=json.load(f)
    return data

def is_grammatically_correct(language_path, seq):
    """
    input
    ----
    seq: a sequence of tokens

    return
    ----
    True: if this sequence respects the grammar
    False: if this sequence does not respect the grammar
    """
    language = json_reader(language_path)
    grammar = {}
    vocab = {}
    for i in language["vocab"]:
        for j in language["vocab"][i]:
            vocab[j] = i
    vocab["<start>"] = "<start>"
    vocab["<end>"] = "<end>"

    for i in language['grammar']:
        grammar[i["LHS"]] = i["RHS"]

    j = 0
    while(j<len(seq)-1):
        try:
            if [vocab[seq[j+1]]] in grammar[vocab[seq[j]]]:
                j = j+1
            elif [vocab[seq[j+1]], vocab[seq[j+2]]] in grammar[vocab[seq[j]]]:
                j = j+2
            elif [vocab[seq[j+1]], vocab[seq[j+2]], vocab[seq[j+3]]] in grammar[vocab[seq[j]]]:
                j = j+3
            else:
                return False
        except:
            return False
    return True

def sim(key, data):
    args = config()
    get_dict = load_data(args)

    seq = ["<start>"]
    for token in data['pred_seq'][:]:
        seq.append(token.decode('utf-8'))

    if "<end>" in seq:
        end_idx = seq.index("<end>")
        seq = seq[:end_idx + 1]
    else:
        return None

    if not is_grammatically_correct("language.json", seq):
        return None

    input_data = {
        "gear_train_sequence": seq,
        "id": 0
    }

    with SuppressPrint():
        res = simulator.run(input_data)

    req_input = data['req_input'][:]
    orig_seq = data['orig_seq'][:]
    pred_seq = seq
    pred_seq_idx = data['pred_seq_idx'][:]
    logits = data['logits'][:]

    return key, res, req_input, orig_seq, pred_seq, pred_seq_idx, logits

if __name__ == "__main__":

    num_files = 74

    for idx in range(10, num_files):

        simulator = Simulator()
        batch_size = 1000

        in_f_name = 'rm_data_' + str(idx) + '.h5'
        out_f_name = 'rm_data_sim_' + str(idx) + '.h5'

        with h5py.File(in_f_name, 'r') as in_f:
            keys = list(in_f.keys())
            data_size = len(keys)

            def process_key(key):
                # Helper function to avoid lambda usage in ProcessPoolExecutor
                return sim(key, in_f[key])

            for i in range(0, data_size, batch_size):
                print("iter ", str(i), " out of ", str(data_size))

                batch_keys = keys[i:i + batch_size]

                with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
                    futures = {executor.submit(process_key, key): key for key in batch_keys}

                with h5py.File(out_f_name, 'a') as output_f:
                    for future in concurrent.futures.as_completed(futures):
                        key = futures[future]
                        try:
                            result = future.result(timeout=1.0)
                            if result is not None:
                                key, res, req_input, orig_seq, pred_seq, pred_seq_idx, logits = result

                                pred_output_motion_speed = res["output_motion_speed"]
                                pred_output_position = res["output_position"]
                                pred_weight = res["weight"]

                                grp = output_f.create_group(key)
                                # Store attr1, attr2, and attr3 as datasets
                                grp.create_dataset('req_input', data=req_input)
                                grp.create_dataset('orig_seq', data=orig_seq)
                                grp.create_dataset('pred_seq', data=pred_seq)  # Convert tensor to numpy for HDF5
                                grp.create_dataset('pred_seq_idx', data=pred_seq_idx)
                                grp.create_dataset('logits', data=logits)  # Convert tensor to numpy for HDF5
                                grp.create_dataset('pred_output_motion_speed', data=pred_output_motion_speed)  # Convert tensor to numpy for HDF5
                                grp.create_dataset('pred_output_position', data=pred_output_position)  # Convert tensor to numpy for HDF5
                                grp.create_dataset('pred_weight', data=pred_weight)  # Convert tensor to numpy for HDF5
                            else:
                                print("failed")       
                        except TimeoutError:
                            print(key)
