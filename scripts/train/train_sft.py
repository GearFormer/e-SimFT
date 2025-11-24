import torch
torch.manual_seed(0)
import numpy as np
import os
import torch.nn as nn
from esimft.utils.data_handle import DataHandler
from esimft.utils.config_file import config
from esimft.model.sft import SFT
from esimft.model.gearformer import GFModel, ObjEncoder, WeightEncoder
from esimft.model.gearformer_simft import GearFormerSimFT
from tqdm import tqdm


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = config()

    config.train_data_path = config.data_sft_train
    config.val_data_path = config.data_sft_val
        
    data_handler = DataHandler(config)

    gfm = GFModel(config, device, encoder_checkpoint_path=config.gearformer_encoder_checkpoint_name, decoder_checkpoint_path=config.gearformer_decoder_checkpoint_name)
    encoder = gfm.encoder
    decoder = gfm.decoder

    if config.sft_mode == "original_req":
        train_loader = data_handler.get_sft_data(config.BS, if_val=False)
        val_loader = data_handler.get_sft_data(config.BS, if_val=True)

        gfm_ft = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)

    elif config.sft_mode == "new_req":
        train_loader = data_handler.get_sft_obj_data(config.BS, if_val=False)
        val_loader = data_handler.get_sft_obj_data(config.BS, if_val=True)        

        new_req_encoder = ObjEncoder(input_size=1, output_size=config.dim).to(device)

        gfm_ft = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, freeze_new_req_encoder=False, device=device)

    elif config.sft_mode == "ric":
        train_loader = data_handler.get_sft_ric_data(config.BS, if_val=False)
        val_loader = data_handler.get_sft_ric_data(config.BS, if_val=True)        

        new_req_encoder = ObjEncoder(input_size=1, output_size=config.dim).to(device)
        weight_encoder = WeightEncoder(input_size=4, output_size=config.dim).to(device)

        gfm_ft = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, weight_encoder=weight_encoder,
            freeze_new_req_encoder=False, freeze_weight_encoder=False, device=device)

    sft = SFT(gfm_ft)

    prev_val_loss = 1e6
    prev_model_state = None

    for epoch in range(config.epoch):

        total_loss_val, val_len = 0, 0
        total_loss_train, train_len = 0, 0

        # train
        for batch_idx, (batch_data) in enumerate(tqdm(train_loader)):

            loss = sft.training_step(inputs=batch_data["inputs"], labels=batch_data["seq"])

            total_loss_train += loss * len(batch_data["seq"])
            train_len += len(batch_data["seq"])

        # validation
        for batch_idx, (batch_data) in enumerate(tqdm(val_loader)):

            loss = sft.validation_step(inputs=batch_data["inputs"], labels=batch_data["seq"])

            total_loss_val += loss * len(batch_data["seq"])
            val_len += len(batch_data["seq"])

        print(f"Epoch: {epoch}, Train loss: {total_loss_train/train_len}, Val loss: {total_loss_val/val_len}") 

        # stop when val loss stops improving and save only the best checkpoint
        if total_loss_val < prev_val_loss:
            prev_model_state = gfm_ft.state_dict()
            prev_val_loss = total_loss_val
        else:
            print("Validation loss stopped improving...")

            model_save_path = os.path.join(config.checkpoint_path, config.sft_model_checkpoint_name)
            torch.save(prev_model_state, model_save_path)

            print(f"Best checkpoints saved at {model_save_path}")

            exit()