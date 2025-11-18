import torch
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from simulator.gear_train_simulator import Simulator
from esimft.utils.processing import SuppressPrint
from .utils.helper import is_grammatically_correct, is_physically_feasible
from concurrent.futures import ThreadPoolExecutor


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

def compute_logprobs(logits, seq):

    seq = seq[:,1:]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=seq.unsqueeze(-1)
    ).squeeze(-1)

    return selected_log_probs

class ActorModel(nn.Module):
    def __init__(self, decoder):
        super(ActorModel, self).__init__()

        self.decoder = decoder

    def forward(self, encoded_input, seq):
        _, (logits, _) = self.decoder(seq, context = encoded_input, return_outputs = True) 

        return logits
    
    def generate(self, prompt, encoded_input):
        ouptut_seq = self.decoder.generate(prompts=prompt, context=encoded_input, seq_len=20, temperature=0.0)
        
        return ouptut_seq

class CriticModel(nn.Module):
    def __init__(self, decoder):
        super(CriticModel, self).__init__()

        self.decoder = decoder

        self.dropout = nn.Dropout(p=0.5)   

        self.final_layer = nn.Linear(20, 1)

    def forward(self, encoded_input, seq):
        _, (logits, _) = self.decoder(seq, context = encoded_input, return_outputs = True) 

        out = compute_logprobs(logits, seq)

        # out = self.dropout(out)

        value = self.final_layer(out)

        value = torch.clamp(value, min=-1, max=0)

        return value.squeeze()
    
class PPO():

    def __init__(self, encoder, obj_encoder, actor, actor_ref, actor_optimizer, args, get_data):
        self.rollout_size = 4
        self.batch_size = args.BS
        self.ppo_mb_size = 16
        self.clip = 0.2
        self.seq_length = 20
        self.beta = 0.1
        self.gamma = 0.95

        self.encoder = encoder
        self.obj_encoder = obj_encoder
        self.actor = actor
        # self.critic = critic
        self.actor_optimizer = actor_optimizer
        # self.critic_optimizer = critic_optimizer
        self.actor_ref = actor_ref
        
        self.get_data = get_data
        self.args = args

    def train(self, req_input, new_obj, seq_output):

        req_input = req_input.to(device0)
        new_obj = new_obj.to(device0).unsqueeze(-1)

        with torch.no_grad():
            reqs_b, seqs_b, log_probs_b, rews_b = self.rollout(req_input, new_obj)

        self.actor.train()

        total_loss = 0
        ppo_mb_size = 8

        for i in range(0, self.batch_size, ppo_mb_size):

            reqs_mb = reqs_b[i : i + ppo_mb_size]
            seqs_mb = seqs_b[i : i + ppo_mb_size]
            log_probs_mb = log_probs_b[i : i + ppo_mb_size]
            rews_mb = rews_b[i : i + ppo_mb_size]

            if reqs_mb.shape[0] == 0:
                continue

            _, (curr_logits, _)  = self.actor(seqs_mb, context = reqs_mb, return_outputs = True) 
            curr_log_prob = compute_logprobs(curr_logits, seqs_mb)

            ratios = torch.exp(curr_log_prob - log_probs_mb).mean(-1)

            surr1 = ratios * rews_mb
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * rews_mb

            _, (ref_logits, _) = self.actor_ref(seqs_mb, context = reqs_mb, return_outputs = True) 
            ref_log_probs = compute_logprobs(ref_logits, seqs_mb)

            ppo_loss = -torch.min(surr1, surr2)
            # ppo_loss = -surr1
            kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)(curr_log_prob, ref_log_probs)

            actor_loss = ppo_loss.mean() + self.beta * kl_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            total_loss += actor_loss.detach().item()

        return total_loss

    def validate(self, req_input, seq_output):

        with torch.no_grad():
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, rew_sum = self.rollout(req_input, seq_output)

        self.actor.eval()
        self.critic.eval()

        curr_logits, curr_log_prob = self.eval_actor(batch_obs, batch_acts)
        curr_log_prob = curr_log_prob.to(device0)
        
        batch_obs = batch_obs.to(device0)
        batch_acts = batch_acts.to(device0)
        batch_log_probs = batch_log_probs.to(device0)
        batch_rtgs = batch_rtgs.to(device0)

        values = self.eval_critic(batch_obs, batch_acts)
        advantages = batch_rtgs - values.detach()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        ratios = torch.exp(curr_log_prob - batch_log_probs).mean(-1)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

        _, (ref_logits, _) = self.actor_ref(batch_acts, context = batch_obs, return_outputs = True) 
        ref_log_probs = compute_logprobs(ref_logits, batch_acts)

        ppo_loss = -torch.min(surr1, surr2)
        kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

        actor_loss = ppo_loss.mean() + self.beta * kl_loss(curr_log_prob, ref_log_probs)

        values = self.eval_critic(batch_obs, batch_acts)

        critic_loss = nn.MSELoss()(values, batch_rtgs)

        return actor_loss.detach().item(), critic_loss.detach().item()
        
    def test_rm(self, req_input, seq_output, rej_seq_output):

        req_input = self.encoder(req_input.to(device0)).to(device1)
        seq_output = seq_output.to(device1)
        rej_seq_output = rej_seq_output.to(device1)

        rew = self.reward_model(req_input, seq_output) + 10.0
        rej_rew = self.reward_model(req_input, rej_seq_output) + 10.0
        
        for i in range(0, len(rew)):
            print(seq_output[i].long().tolist())
            print(rej_seq_output[i].long().tolist())
            print(rew[i].item(), rej_rew[i].item())
            input()

    def rollout(self, reqs, new_objs):

        rollout_size = reqs.shape[0]
        reqs_aug = self.encoder(reqs) + self.obj_encoder(new_objs)

        seq = ["<start>"]
        prompt = torch.zeros((rollout_size, len(seq))).long().to(device0)
        for i in range(self.rollout_size):
            for j in range(len(seq)):
                prompt[i, j] = 0
        
        zeros = torch.zeros(rollout_size, 1).to(device0)
        seqs_samp = self.actor.generate(prompts=prompt, context=reqs_aug, seq_len=20, temperature=1.0)
        seqs_samp = torch.cat((zeros, seqs_samp), dim=1).long()
        
        _, (logits, _) = self.actor(seqs_samp.long(), context = reqs_aug, return_outputs = True) 
        log_probs = compute_logprobs(logits, seqs_samp)

        rews = self.compute_rews(new_objs, seqs_samp)

        return reqs_aug, seqs_samp, log_probs, rews

    def run_simulator(self, input_data):

        obj_name = "price"

        seq = input_data["gear_train_sequence"]
        simulator = Simulator()

        if not is_grammatically_correct(self.args, seq):
            # print("grammar")
            rew = -1
        elif not is_physically_feasible(seq, self.args.catalogue_path):
            # print("physics")
            rew = -1
        else:
            try:
                with SuppressPrint():
                    actual_obj = simulator.run(input_data)[obj_name]
                target_obj = input_data["target_obj"]
                if actual_obj <= target_obj:
                    rew = 1
                else:
                    x = actual_obj - target_obj
                    rew = 1 - (2 * x) / (1 + x)
                    # rew = -1
            except:
                rew = -1

        return rew

    def compute_rews(self, objs, seqs):

        num_threads = 32
        sim_inputs = []

        for i in range(0, len(seqs)):

            seq_i = seqs[i].to("cpu").long().tolist()
            if 51 in seq_i:
                seq_i = seq_i[:seq_i.index(51)+1]

            seq = []
            for j in range(0, len(seq_i)):
                seq.append(self.get_data.inx2name(seq_i[j]))

            sim_inputs.append({
                "id": i,
                "gear_train_sequence": seq,
                "target_obj": objs[i].detach().item()
            })

        with ThreadPoolExecutor(max_workers=num_threads) as executor, SuppressPrint():
            results = list(executor.map(self.run_simulator, sim_inputs))

        rews = torch.tensor(results, dtype=torch.float32).to(device0)

        return rews

    def compute_rtgs(self, batch_rews):

        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.stack(batch_rtgs)

        return batch_rtgs

    def eval_actor(self, obs, acts):

        logits = self.actor(obs, acts)

        log_prob = compute_logprobs(logits, acts)

        return logits, log_prob
    
    def get_new_acts(self, logits):

        next_token_p = torch.nn.functional.softmax(logits, dim=-1)
        next_token_idx = torch.argmax(next_token_p, dim=-1)
        zeros = torch.zeros(next_token_idx.shape[0], 1).to(device0)
        new_acts = torch.cat((zeros, next_token_idx), dim=1)

        return new_acts

    def get_new_acts2(self, batch_prompt, encoded_input):

        # new_acts = self.actor.generate(batch_prompt, encoded_input)
        new_acts = self.actor.generate(prompts=batch_prompt, context=encoded_input, seq_len=20, temperature=0.0)
        zeros = torch.zeros(new_acts.shape[0], 1).to(device0)
        new_acts = torch.cat((zeros, new_acts), dim=1)

        return new_acts
    
    def eval_critic(self, obs, acts):

        V = self.critic(obs, acts)

        return V

