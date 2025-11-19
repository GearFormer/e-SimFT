import torch
torch.manual_seed(0)
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
from esimft.utils.data_handle import DataHandler
from esimft.utils.config_file import config
from esimft.model.sft import SFT
from esimft.model.gearformer import GFModel
from tqdm import tqdm


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = config()

    if config.sft_mode == "original_req":
        config.train_data_path = config.sft_data.replace(".pkl", f"_{config.req_name}_train.pkl")
        config.val_data_path = config.sft_data.replace(".pkl", f"_{config.req_name}_val.pkl")
    elif config.sft_mode == "new_req":
        config.train_data_path = config.sft_data.replace(".pkl", f"_new_obj_train.pkl")
        config.val_data_path = config.sft_data.replace(".pkl", f"_new_obj_val.pkl")
    elif config.sft_mode == "ric":
        config.train_data_path = config.ric_train_data
        config.val_data_path = config.ric_val_data
        
    data_handler = DataHandler(config)

    gfm = GFModel(config, device)

    if config.sft_mode == "original_req":
        train_loader = data_handler.get_sft_data(config.BS, if_val=False)
        val_loader = data_handler.get_sft_data(config.BS, if_val=True)
        sft = SFT(config, gfm, freeze_encoder=True)
    elif config.sft_mode == "new_req":
        train_loader = data_handler.get_sft_obj_data(config.BS, if_val=False)
        val_loader = data_handler.get_sft_obj_data(config.BS, if_val=True)        
        sft = SFT(config, gfm, freeze_encoder=True, fit_new_req=True)
    elif config.sft_mode == "ric":
        train_loader = data_handler.get_sft_ric_data(config.BS, if_val=False)
        val_loader = data_handler.get_sft_ric_data(config.BS, if_val=True)        
        sft = SFT(config, gfm, freeze_encoder=True, fit_new_req=True, fit_req_weights=True)

    prev_val_loss = 1e6
    prev_encoder_state = None
    prev_decoder_state = None

    for epoch in range(config.epoch):

        total_loss_val, val_len = 0, 0
        total_loss_train, train_len = 0, 0
            
        # train
        for batch_idx, (req_input, chosen_seq, new_req_list, weights) in enumerate(tqdm(train_loader)):
            req_input = req_input.to(device)
            seq = chosen_seq.to(device)

            if config.sft_mode == "new_req":
                if config.req_name == "price":
                    new_req_list = new_req_list[:,0].unsqueeze(-1).to(device)
                elif config.req_name == "bb":
                    new_req_list = new_req_list[:,1].unsqueeze(-1).to(device)
                weights = None

            elif config.sft_mode == "ric":
                new_req_list = new_req_list.to(device)
                weights = weights.to(device)

            else:
                new_req_list = None
                weights = None

            loss = sft.training_step(input_data=req_input, target=seq, ignore_index=config.output_size-1, new_req_list=new_req_list, weights=weights)

            total_loss_train += loss * len(req_input)
            train_len += len(req_input)

        # validation
        for batch_idx, (req_input, chosen_seq, new_req_list, weights) in enumerate(tqdm(val_loader)):
            req_input = req_input.to(device)
            seq = chosen_seq.to(device)

            if config.sft_mode == "new_req":
                if config.req_name == "price":
                    new_req_list = new_req_list[:,0].unsqueeze(-1).to(device)
                elif config.req_name == "bb":
                    new_req_list = new_req_list[:,1].unsqueeze(-1).to(device)
                weights = None

            elif config.sft_mode == "ric":
                new_req_list = new_req_list.to(device)
                weights = weights.to(device)
                
            else:
                new_req_list = None
                weights = None

            loss = sft.validation_step(input_data=req_input, target=seq, ignore_index=config.output_size-1, new_req_list=new_req_list)

            total_loss_val += loss * len(req_input)
            val_len += len(req_input)

        print(f"Epoch: {epoch}, Train loss: {total_loss_train/train_len}, Val loss: {total_loss_val/val_len}") 

        # stop when val loss stops improving and save only the best checkpoint
        if total_loss_val < prev_val_loss:
            prev_encoder_state = sft.encoder.state_dict()
            prev_decoder_state = sft.decoder.state_dict()
            prev_val_loss = total_loss_val
        else:
            print("Validation loss stopped improving...")

            encoder_checkpoint_name = "SFT_" + config.req_name + "_encoder.dict"
            decoder_checkpoint_name = "SFT_" + config.req_name + "_decoder.dict"

            encoder_save_path =  os.path.join(config.checkpoint_path, encoder_checkpoint_name)
            decoder_save_path =  os.path.join(config.checkpoint_path, decoder_checkpoint_name)

            torch.save(prev_encoder_state, encoder_save_path)
            torch.save(prev_decoder_state, decoder_save_path)

            print(f"Best checkpoints saved at {encoder_save_path} and {decoder_save_path}.")
            print("Run finished.")

            exit()