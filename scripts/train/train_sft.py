import random
random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import os
import torch.optim as optim
import torch.nn as nn
from .utils.data_handle import load_data
from .utils.config_file import config
from .load_model import loading_model
from .sft import train_sft, val_sft
from tqdm import tqdm

if __name__ == "__main__":
    
    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = config()    
    model = args.model_name
    get_data = load_data(args)
    req_input_size = 8

    train_loader = get_data.get_sft_data(args.BS, if_val=False)
    val_loader = get_data.get_sft_data(args.BS, if_val=True)
    
    encoder, decoder = loading_model(args, req_input_size, get_data.output_size, get_data.max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    encoder = encoder.to(device0)
    encoder.eval()
    decoder = decoder.to(device0)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    prev_val_loss = 1e6
    prev_state_dict = None

    for epoch in range(args.epoch):

        total_loss_val, val_len = 0, 0
        total_loss_train, train_len = 0, 0
            
        for batch_idx, (req_input, chosen_seq) in enumerate(tqdm(train_loader)):
            req_input = req_input.to(device0)
            seq = chosen_seq.to(device0)

            encoded_input = encoder(req_input)

            loss = train_sft(encoded_input, seq, decoder, decoder_optimizer)
            total_loss_train += loss * len(req_input)
            train_len += len(req_input)

        for batch_idx, (req_input, chosen_seq) in enumerate(tqdm(val_loader)):
            req_input = req_input.to(device0)
            seq = chosen_seq.to(device0)
            
            encoded_input = encoder(req_input)

            loss = val_sft(encoded_input, seq, decoder)
            total_loss_val += loss * len(req_input)
            val_len += len(req_input)

        print(epoch, "train Loss:", total_loss_train/train_len) 
        print(epoch, "val Loss:", total_loss_val/val_len)

        if total_loss_val < prev_val_loss:
            prev_state_dict = decoder.state_dict()
            prev_val_loss = total_loss_val
        else:
            decoder_checkpoint_name = "SFT_" + args.req_name + "_decoder.dict"
            torch.save(prev_state_dict, os.path.join(args.checkpoint_path, decoder_checkpoint_name))
            exit()