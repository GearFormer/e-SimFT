import torch
import numpy as np
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from esimft.model.functions import compute_logprobs
from esimft.model.gearformer import ObjEncoder
from typing import Tuple
import copy
import os


class DPOLoss(nn.Module):

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ):

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits)

        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

class DPO:
    def __init__(self, model):

        self.model = model
        self.model_ref = copy.deepcopy(model)

        self.dpo_loss = DPOLoss(beta=0.1)

    def training_step(self, inputs, preferred_labels, rejected_labels):
        
        self.model.optimizer.zero_grad()

        preferred_logits = self.model(inputs, preferred_labels)
        rejected_logits = self.model(inputs, rejected_labels)

        preferred_labels = preferred_labels.long().to(preferred_logits.device)
        rejected_labels = rejected_labels.long().to(rejected_logits.device)

        preferred_logp = compute_logprobs(preferred_logits, preferred_labels[:,1:])
        rejected_logp = compute_logprobs(rejected_logits, rejected_labels[:,1:])

        with torch.no_grad():
            preferred_logits_ref = self.model_ref(inputs, preferred_labels)
            rejected_logits_ref = self.model_ref(inputs, rejected_labels)

            preferred_logp_ref = compute_logprobs(preferred_logits_ref, preferred_labels[:,1:])
            rejected_logp_ref = compute_logprobs(rejected_logits_ref, rejected_labels[:,1:])

        loss, preferred_rewards, rejected_rewards = self.dpo_loss(preferred_logp, rejected_logp, preferred_logp_ref, rejected_logp_ref)
        
        loss = loss.mean(-1).mean()
        preferred_rewards = preferred_rewards.mean(-1).mean()
        rejected_rewards = rejected_rewards.mean(-1).mean()

        loss.backward()

        self.model.optimizer.step()

        return loss.item(), preferred_rewards.item(), rejected_rewards.item()

    def validation_step(self, inputs, preferred_labels, rejected_labels):
        
        self.model.eval()

        with torch.no_grad():

            preferred_logits = self.model(inputs, preferred_labels)
            rejected_logits = self.model(inputs, rejected_labels)

            preferred_labels = preferred_labels.long().to(preferred_logits.device)
            rejected_labels = rejected_labels.long().to(rejected_logits.device)

            preferred_logp = compute_logprobs(preferred_logits, preferred_labels[:,1:])
            rejected_logp = compute_logprobs(rejected_logits, rejected_labels[:,1:])

            preferred_logits_ref = self.model_ref(inputs, preferred_labels)
            rejected_logits_ref = self.model_ref(inputs, rejected_labels)

            preferred_logp_ref = compute_logprobs(preferred_logits_ref, preferred_labels[:,1:])
            rejected_logp_ref = compute_logprobs(rejected_logits_ref, rejected_labels[:,1:])

            loss, preferred_rewards, rejected_rewards = self.dpo_loss(preferred_logp, rejected_logp, preferred_logp_ref, rejected_logp_ref)
        
            loss = loss.mean(-1).mean()
            preferred_rewards = preferred_rewards.mean(-1).mean()
            rejected_rewards = rejected_rewards.mean(-1).mean()

        return loss.item(), preferred_rewards.item(), rejected_rewards.item()
