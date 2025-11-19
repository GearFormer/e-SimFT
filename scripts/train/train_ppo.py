import torch
torch.manual_seed(0)
import numpy as np
import os
import torch.optim as optim
from esimft.utils.data_handle import DataHandler
from esimft.utils.config_file import config
from esimft.model.ppo import PPO
from esimft.model.gearformer import GFModel, ObjEncoder
from esimft.model.gearformer_simft import GearFormerSimFT
from esimft.model.functions import GearFormerReward
from tqdm import tqdm


device = torch.device("cuda")

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

    gfm_rew = GearFormerReward(config)

    ppo = PPO(actor=sft_model, rews_fn=gfm_rew.gearformer_sim_rews, 
        temperature=config.ppo_temperature, beta=config.ppo_beta, clip=config.ppo_clip, mb_size=config.ppo_mb_size)

    prev_val_loss = 1e6
    prev_model_state = None

    for epoch in range(config.epoch):

        total_loss_train, train_len = 0, 0

        for batch_idx, (batch_data) in enumerate(tqdm(train_loader)):

            inputs = batch_data["inputs"]
            labels = batch_data["chosen_seq"].long().to(device)

            loss = ppo.training_step(inputs, labels)

            total_loss_train += loss * len(labels)
            train_len += len(labels)

        print(f"Epoch: {epoch}, Train loss: {total_loss_train/train_len}") 
        print()
        
        checkpoint_folder = os.path.join(config.checkpoint_path, config.ppo_model_checkpoint_folder)
        os.makedirs(checkpoint_folder, exist_ok=True)
        torch.save(sft_model.state_dict(), os.path.join(checkpoint_folder, f"epoch_{epoch}.dict"))
