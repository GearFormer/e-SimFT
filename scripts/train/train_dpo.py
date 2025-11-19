import torch
torch.manual_seed(0)
import numpy as np
import os
import torch.optim as optim
from esimft.utils.data_handle import DataHandler
from esimft.utils.config_file import config
from esimft.model.dpo import DPO
from esimft.model.gearformer import GFModel, ObjEncoder
from esimft.model.gearformer_simft import GearFormerSimFT
from tqdm import tqdm

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config()

    config.train_data_path = config.data_pref_train
    config.val_data_path = config.data_pref_val

    data_handler = DataHandler(config)

    train_loader = data_handler.get_pref_obj_data(config.BS, if_val=False)
    val_loader = data_handler.get_pref_obj_data(config.BS, if_val=True)

    gfm = GFModel(config, device)
    encoder = gfm.encoder
    decoder = gfm.decoder
    new_req_encoder = ObjEncoder(input_size=1, output_size=config.dim).to(device)

    sft_model = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=new_req_encoder, device=device)
    sft_model.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.sft_model_checkpoint_name), map_location=device))

    dpo = DPO(sft_model)

    prev_val_loss = 1e6
    prev_model_state = None

    for epoch in range(config.epoch):
        
        total_loss_val, val_len = 0, 0
        total_loss_train, train_len = 0, 0
        total_preferred_rewards, total_reject_rewards = 0, 0

        for batch_idx, (batch_data) in enumerate(tqdm(train_loader)):

            inputs = batch_data["inputs"]
            preferred_labels = batch_data["chosen_seq"]
            reject_labels = batch_data["rejected_seq"]

            loss, preferred_rewards, rejected_rewards = dpo.training_step(inputs, preferred_labels, reject_labels)

            total_loss_train += loss * len(batch_data["chosen_seq"])
            train_len += len(batch_data["chosen_seq"])

            total_preferred_rewards += preferred_rewards
            total_reject_rewards += rejected_rewards

        for batch_idx, (batch_data) in enumerate(tqdm(val_loader)):

            inputs = batch_data["inputs"]
            preferred_labels = batch_data["chosen_seq"]
            reject_labels = batch_data["rejected_seq"]

            loss, preferred_rewards, rejected_rewards = dpo.validation_step(inputs, preferred_labels, reject_labels)

            total_loss_val += loss * len(batch_data["chosen_seq"])
            val_len += len(batch_data["chosen_seq"])

        print(f"Epoch: {epoch}, Train loss: {total_loss_train/train_len}, Val loss: {total_loss_val/val_len}") 
        print(f"Chosen rewards: {total_preferred_rewards/train_len}, Rejected rewards: {total_reject_rewards/train_len}, Diff: {(total_preferred_rewards-total_reject_rewards)/train_len}") 
        print()

        # save at every epoch since we need to use the simulation-based eval metric to decide the best model
        checkpoint_folder = os.path.join(config.checkpoint_path, config.dpo_model_checkpoint_folder)
        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(sft_model.state_dict(), os.path.join(checkpoint_folder, f"epoch_{epoch}.dict"))
