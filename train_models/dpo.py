import torch
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

    
class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss module: https://arxiv.org/abs/2305.18290
    Simply stated from the paper:

        Intuitively, the DPO update increases the relative log probability of preferred to dispreferred responses,
        but it incorporates a dynamic, per-example importance weight that prevents
        the model degeneration that we find occurs with a naive probability ratio objective.

    Based on the implementation in HF's TRL library:
    https://github.com/huggingface/trl/blob/5d1deb1445828cfd0e947cb3a7925b1c03a283fc/trl/trainer/dpo_trainer.py#L844

    DPO retains similarities to PPO (https://arxiv.org/abs/2009.01325), where it optimizes a policy
    (language) model to align with human preferences, and regularizes the loss function using a baseline
    reference (the frozen, initial language model) to prevent over-fitting to the preference dataset.
    It differs from PPO by optimizing the policy model directly using labelled preference data, rather
    than using an additional reward model to provide feedback.
    This significantly simplifies training and reduces compute overhead.

    Args:
        beta (float): Temperature parameter for the DPO loss, typically in the range of 0.1 to 0.5. Default is 0.1.
        label_smoothing (float): Parameter encoding uncertainty about the labels. Default is 0.
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps (torch.Tensor): Log probabilities of the policy model
                for the chosen responses. Shape: (batch_size)
            policy_rejected_logps (torch.Tensor): Log probabilities of the policy model
                for the rejected responses. Shape: (batch_size)
            reference_chosen_logps (torch.Tensor): Log probabilities of the reference model
                for the chosen responses. Shape: (batch_size)
            reference_rejected_logps (torch.Tensor): Log probabilities of the reference model
                for the rejected responses. Shape: (batch_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors:
                - losses: The DPO loss for each example in the batch.
                - chosen_rewards: Rewards for the chosen responses.
                - rejected_rewards: Rewards for the rejected responses.

        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        loss = -F.logsigmoid(self.beta * logits)

        chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards

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

def finetune_DPO(input_data, chosen_seq, reject_seq, decoder, decoder_optimizer, chosen_seq_ref, reject_seq_ref, decoder_ref):
    
    decoder.train()
    decoder_optimizer.zero_grad()

    _, (chosen_logits_ft, _) = decoder(chosen_seq.long(), context = input_data, return_outputs = True) 
    chosen_logp_ft = compute_logprobs(chosen_logits_ft, chosen_seq.long())
    _, (reject_logits_ft, _) = decoder(reject_seq.long(), context = input_data, return_outputs = True) 
    reject_logp_ft = compute_logprobs(reject_logits_ft, reject_seq.long())

    _, (chosen_logits_ref, _) = decoder_ref(chosen_seq_ref.long(), context = input_data, return_outputs = True) 
    chosen_logp_ref = compute_logprobs(chosen_logits_ref, chosen_seq_ref.long())
    _, (reject_logits_ref, _) = decoder_ref(reject_seq_ref.long(), context = input_data, return_outputs = True) 
    reject_logp_ref = compute_logprobs(reject_logits_ref, reject_seq_ref.long())

    dpo_loss = DPOLoss(beta=0.1)
    loss, chosen_rewards, rejected_rewards = dpo_loss(chosen_logp_ft, reject_logp_ft, chosen_logp_ref, reject_logp_ref)
    loss = loss.mean(-1).mean()
    chosen_rewards = chosen_rewards.mean(-1).mean()
    rejected_rewards = rejected_rewards.mean(-1).mean()

    # output_size=53
    # target = chosen_seq.long()[:, 1:]
    # ce_loss = torch.nn.functional.cross_entropy(
    #         rearrange(chosen_logits_ft, 'b n c -> b c n'),
    #         target,
    #         ignore_index = output_size-1,
    #         reduction = "none"
    # )
    # ce_loss = ce_loss.mean(-1).mean()

    tota_loss = loss

    tota_loss.backward()
    
    decoder_optimizer.step()

    return loss.item(), chosen_rewards.item(), rejected_rewards.item()

def val_DPO(input_data, chosen_seq, reject_seq, decoder, chosen_seq_ref, reject_seq_ref, decoder_ref):
    
    decoder.eval()

    with torch.no_grad():

        _, (chosen_logits_ft, _) = decoder(chosen_seq.long(), context = input_data, return_outputs = True) 
        chosen_logp_ft = compute_logprobs(chosen_logits_ft, chosen_seq.long())
        _, (reject_logits_ft, _) = decoder(reject_seq.long(), context = input_data, return_outputs = True) 
        reject_logp_ft = compute_logprobs(reject_logits_ft, reject_seq.long())

        _, (chosen_logits_ref, _) = decoder_ref(chosen_seq_ref.long(), context = input_data, return_outputs = True) 
        chosen_logp_ref = compute_logprobs(chosen_logits_ref, chosen_seq_ref.long())
        _, (reject_logits_ref, _) = decoder_ref(reject_seq_ref.long(), context = input_data, return_outputs = True) 
        reject_logp_ref = compute_logprobs(reject_logits_ref, reject_seq_ref.long())

        dpo_loss = DPOLoss(beta=0.1)
        loss, chosen_rewards, rejected_rewards = dpo_loss(chosen_logp_ft, reject_logp_ft, chosen_logp_ref, reject_logp_ref)
        
        loss = loss.mean(-1).mean()
        chosen_rewards = chosen_rewards.mean(-1).mean()
        rejected_rewards = rejected_rewards.mean(-1).mean()

        # output_size=53
        # target = chosen_seq.long()[:, 1:]
        # ce_loss = torch.nn.functional.cross_entropy(
        #         rearrange(chosen_logits_ft, 'b n c -> b c n'),
        #         target,
        #         ignore_index = output_size-1,
        #         reduction = "none"
        # )
        # ce_loss = ce_loss.mean(-1).mean()

        tota_loss = loss

        return tota_loss.item(), chosen_rewards.item(), rejected_rewards.item()
 