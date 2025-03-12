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

def train_sft(input_data, seq, decoder, decoder_optimizer, obj, obj_encoder, encoder_optimizer):
    
    obj_encoder.train()
    decoder.train()
    decoder_optimizer.zero_grad()
    encoder_optimizer.zero_grad()

    obj = obj_encoder(obj)
    aug_input = input_data + obj    

    _, (logits, _) = decoder(seq.long(), context = aug_input, return_outputs = True) 
    
    # target = seq.long()[:, 1:]
    # loss = torch.nn.functional.cross_entropy(
    #         rearrange(logits, 'b n c -> b c n'),
    #         target,
    #         ignore_index = output_size-1,
    #         reduction = "none"
    # )
    # loss = loss.mean(-1).mean()

    log_prob = compute_logprobs(logits, seq.long())
    loss = -log_prob.mean(-1).mean()

    loss.backward()
    
    decoder_optimizer.step()
    encoder_optimizer.step()

    return loss.item()

def val_sft(input_data, seq, decoder, obj, obj_encoder):
    
    obj_encoder.eval()
    decoder.eval()

    obj = obj_encoder(obj)
    aug_input = input_data + obj
        
    with torch.no_grad():

        _, (logits, _) = decoder(seq.long(), context = aug_input, return_outputs = True) 
        
        # target = seq.long()[:, 1:]
        # loss = torch.nn.functional.cross_entropy(
        #         rearrange(logits, 'b n c -> b c n'),
        #         target,
        #         ignore_index = output_size-1,
        #         reduction = "none"
        # )
        # loss = loss.mean(-1).mean()

        log_prob = compute_logprobs(logits, seq.long())
        loss = -log_prob.mean(-1).mean()

        return loss.item()
 