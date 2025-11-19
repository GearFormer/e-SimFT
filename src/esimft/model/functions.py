import torch
import torch.nn.functional as F
from einops import rearrange
from esimft.utils.processing import SuppressPrint
from concurrent.futures import ThreadPoolExecutor
from esimft.gear_train_simulator.simulator import Simulator
from esimft.utils.gearformer.helper import is_grammatically_correct, is_physically_feasible
from esimft.utils.data_handle import DataHandler
from esimft.utils.config_file import config

def compute_logprobs(logits, target):

    log_probs = F.log_softmax(logits, dim=-1)

    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=target.unsqueeze(-1)
    ).squeeze(-1)

    return selected_log_probs


def compute_cross_entropy_loss(logits, target, ignore_index):

    loss = torch.nn.functional.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index,
            reduction = "none"
    )
    loss = loss.mean(-1).mean()

    return loss






class GearFormerReward:

    def __init__(self, config):
        self.config = config
        self.simulator = Simulator()


    def calculate_volume(self, min_corner, max_corner):
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

    def run_simulator(self, input_data):

        seq = input_data["gear_train_sequence"]

        if not is_grammatically_correct(self.config, seq):
            rew = -1
        elif not is_physically_feasible(seq, self.config.catalogue_path):
            rew = -1
        else:
            try:
                with SuppressPrint():
                    res = self.simulator.run(input_data)

                if self.config.req_name == "bb":
                    actual_obj = self.calculate_volume(res["bounding_box_min"], res["bounding_box_max"])
                elif self.config.req_name == "price":
                    actual_obj = res["price"]

                target_obj = input_data["target_obj"]
                
                if actual_obj <= target_obj:
                    rew = 1
                else:
                    x = actual_obj - target_obj
                    rew = 1 - (2 * x) / (1 + x)
            except:
                rew = -1

        return rew

    def gearformer_sim_rews(self, inputs, labels):

        data_handler = DataHandler(self.config)

        target_values = inputs[1]

        num_threads = 32
        sim_inputs = []

        for i in range(0, len(labels)):

            seq_i = labels[i].long().tolist()

            if 51 in seq_i:
                seq_i = seq_i[:seq_i.index(51)+1]
            else:
                seq_i[-1] = 51

            seq = []
            for j in range(0, len(seq_i)):
                seq.append(data_handler.inx2name(seq_i[j]))

            sim_inputs.append({
                "id": i,
                "gear_train_sequence": seq,
                "target_obj": target_values[i].item()
            })

        with ThreadPoolExecutor(max_workers=self.config.num_threads_sim) as executor, SuppressPrint():
            results = list(executor.map(self.run_simulator, sim_inputs))

        rews = torch.tensor(results, dtype=torch.float32).to(labels.device)

        return rews