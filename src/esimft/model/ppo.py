import torch
import numpy as np
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from esimft.model.functions import compute_logprobs
import copy


class PPO:

    def __init__(self, actor, rews_fn, temperature=1.0, beta=0.1, clip=0.2, mb_size=8):
        
        self.actor = actor
        self.actor_ref = copy.deepcopy(actor)
        self.rews_fn = rews_fn

        self.temperature = temperature
        self.beta = beta
        self.clip = clip
        self.mb_size = mb_size
        self.rollout_size = None

    def training_step(self, inputs, labels):

        self.rollout_size = labels.shape[0]

        with torch.no_grad():
            sampled, log_probs, rews = self.rollout(inputs)

        self.actor.train()

        total_loss = 0

        for i in range(0,  self.rollout_size, self.mb_size):
            self.actor.optimizer.zero_grad()

            if isinstance(inputs, torch.Tensor):
                inputs_mb = inputs[i : i + self.mb_size]
            else: # inputs is an aggregate of tensors
                inputs_mb = []
                for inputs_k in inputs:
                    inputs_mb.append(inputs_k[i : i + self.mb_size])

            labels_mb = labels[i : i + self.mb_size]
            log_probs_mb = log_probs[i : i + self.mb_size]
            rews_mb = rews[i : i + self.mb_size]

            curr_logits  = self.actor(inputs=inputs_mb, labels=labels_mb) 

            curr_log_prob = compute_logprobs(curr_logits, labels_mb[:,1:])

            ratios = torch.exp(curr_log_prob - log_probs_mb).mean(-1)
            surr1 = ratios * rews_mb
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * rews_mb

            ref_logits = self.actor_ref(inputs=inputs_mb, labels=labels_mb) 
            ref_log_probs = compute_logprobs(ref_logits, labels_mb[:,1:])

            ppo_loss = -torch.min(surr1, surr2)
            kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)(curr_log_prob, ref_log_probs)

            loss = ppo_loss.mean() + self.beta * kl_loss
            
            loss.backward()

            self.actor.optimizer.step()

            total_loss += loss.detach().item()

        return total_loss

    def rollout(self, inputs, start_token=0):

        prompts = torch.full(
            (self.rollout_size, 1),
            fill_value=start_token,
            dtype=torch.long,
            device=self.actor.device,
        )

        sampled = self.actor.generate(inputs=inputs, prompts=prompts, temperature=self.temperature)
        sampled = torch.cat((prompts, sampled), dim=1)
        
        logits = self.actor(inputs=inputs, labels=sampled) 
        log_probs = compute_logprobs(logits, sampled[:,1:])

        rews = self.rews_fn(inputs, sampled)

        return sampled, log_probs, rews


# class ActorModel(nn.Module):
#     def __init__(self, decoder):
#         super(ActorModel, self).__init__()

#         self.decoder = decoder

#     def forward(self, encoded_input, seq):
#         _, (logits, _) = self.decoder(seq, context = encoded_input, return_outputs = True) 

#         return logits
    
#     def generate(self, prompt, encoded_input):
#         ouptut_seq = self.decoder.generate(prompts=prompt, context=encoded_input, seq_len=20, temperature=0.0)
        
#         return ouptut_seq

# class CriticModel(nn.Module):
#     def __init__(self, decoder):
#         super(CriticModel, self).__init__()

#         self.decoder = decoder

#         self.dropout = nn.Dropout(p=0.5)   

#         self.final_layer = nn.Linear(20, 1)

#     def forward(self, encoded_input, seq):
#         _, (logits, _) = self.decoder(seq, context = encoded_input, return_outputs = True) 

#         target = seq[:,1:]

#         out = compute_logprobs(logits, target)

#         value = self.final_layer(out)

#         value = torch.clamp(value, min=-1, max=0)

#         return value.squeeze()
    