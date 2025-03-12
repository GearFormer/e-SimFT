import random
random.seed(0)
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import os
import torch.optim as optim
from .utils.data_handle import load_data
from .utils.config_file import config
from .load_model import loading_model
from .dpo import finetune_DPO, val_DPO
from .transformers import ObjEncoder
from tqdm import tqdm

if __name__ == "__main__":
    
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    args = config()    
    model = args.model_name
    get_data = load_data(args)
    req_input_size = 8

    train_loader = get_data.get_pref_obj_data(args.BS, if_val=False)
    val_loader = get_data.get_pref_obj_data(args.BS, if_val=True)
    
    encoder, decoder = loading_model(args, req_input_size, get_data.output_size, get_data.max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    encoder = encoder.to(device0)
    encoder.eval()
    decoder = decoder.to(device0)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    _, decoder_ref = loading_model(args, req_input_size, get_data.output_size, get_data.max_length)
    decoder_ref.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    decoder_ref = decoder_ref.to(device0)
    decoder_ref.eval()

    obj_encoder = ObjEncoder(input_size=1, output_size=args.dim)
    obj_encoder_name = "/app/train_models/models/SFT_" + args.req_name + "_new_encoder.dict"
    obj_encoder.load_state_dict(torch.load(obj_encoder_name))
    obj_encoder = obj_encoder.to(device0)
    obj_encoder.eval()

    total_loss_val, val_len = 0, 0
    total_loss_train, train_len = 0, 0

    total_chosen_rewards, total_reject_rewards = 0, 0

    prev_val_loss = 1e6
    decoder_state_dict = None
    encoder_state_dict = None
    
    for epoch in range(args.epoch):
            
        for batch_idx, (req_input, chosen_seq, reject_seq, mid_value) in enumerate(tqdm(train_loader)):
            req_input = encoder(req_input.to(device0))
            obj_input = obj_encoder(mid_value.unsqueeze(-1).to(device0))
            aug_input = req_input + obj_input

            chosen_seq_ft = chosen_seq.to(device0)
            reject_seq_ft = reject_seq.to(device0)
            
            chosen_seq_ref = chosen_seq.to(device0)
            reject_seq_ref = reject_seq.to(device0)

            loss, chosen_rewards, rejected_rewards = finetune_DPO(aug_input, chosen_seq_ft, reject_seq_ft, decoder, decoder_optimizer, 
                                                                  chosen_seq_ref, reject_seq_ref, decoder_ref)
            total_loss_train += loss * len(req_input)
            train_len += len(req_input)

            total_chosen_rewards += chosen_rewards
            total_reject_rewards += rejected_rewards

        for batch_idx, (req_input, chosen_seq, reject_seq, mid_value) in enumerate(tqdm(val_loader)):
            req_input = encoder(req_input.to(device0))
            obj_input = obj_encoder(mid_value.unsqueeze(-1).to(device0))
            aug_input = req_input + obj_input

            chosen_seq_ft = chosen_seq.to(device0)
            reject_seq_ft = reject_seq.to(device0)
            
            chosen_seq_ref = chosen_seq.to(device0)
            reject_seq_ref = reject_seq.to(device0)
            
            loss, chosen_rewards, rejected_rewards = val_DPO(aug_input, chosen_seq_ft, reject_seq_ft, decoder, 
                                                             chosen_seq_ref, reject_seq_ref, decoder_ref)
            total_loss_val += loss * len(req_input)
            val_len += len(req_input)

        val_loss = total_loss_val/val_len
        print(epoch, "train Loss:", total_loss_train/train_len) 
        print(epoch, "val Loss:", val_loss)
        print(epoch, "total chosen reward:", total_chosen_rewards/train_len)
        print(epoch, "total reject reward:", total_reject_rewards/train_len)
        print(epoch, "diff", (total_chosen_rewards-total_reject_rewards)/train_len)

        decoder_checkpoint_name = "DPO_" + args.req_name + "_" + str(epoch) + "_decoder.dict"
        torch.save(decoder.state_dict(), os.path.join(args.checkpoint_path, decoder_checkpoint_name))