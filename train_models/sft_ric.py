import torch
import random
random.seed(0)
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F


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

def train_sft(input_data, seq, decoder, decoder_optimizer, obj_encoder_optimizer, w_encoder_optimizer):

    decoder.train()
    decoder_optimizer.zero_grad()
    obj_encoder_optimizer.zero_grad()
    w_encoder_optimizer.zero_grad()

    _, (logits, _) = decoder(seq.long(), context = input_data, return_outputs = True) 
    
    log_prob = compute_logprobs(logits, seq.long())
    loss = -log_prob.mean(-1).mean()

    loss.backward()
    
    decoder_optimizer.step()
    obj_encoder_optimizer.step()
    w_encoder_optimizer.step()

    return loss.item()

def val_sft(input_data, seq, decoder):
    
    decoder.eval()
        
    with torch.no_grad():

        _, (logits, _) = decoder(seq.long(), context = input_data, return_outputs = True) 

        log_prob = compute_logprobs(logits, seq.long())
        loss = -log_prob.mean(-1).mean()

        return loss.item()
 