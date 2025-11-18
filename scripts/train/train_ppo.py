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
from .ppo_bb import PPO as PPO_bb
from .ppo_price import PPO as PPO_price
from .transformers import ObjEncoder
from tqdm import tqdm


device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

if __name__ == "__main__":
    
    args = config()    
    model = args.model_name
    get_data = load_data(args)
    req_input_size = 8

    train_loader = get_data.get_pref_obj_data(args.BS, if_val=False)
    val_loader = get_data.get_pref_obj_data(args.BS, if_val=True)
    
    encoder, actor = loading_model(args, req_input_size, get_data.output_size, get_data.max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)))
    encoder = encoder.to(device0)
    encoder.eval()  
    
    actor.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    actor_model = actor.to(device0)
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=args.lr)

    _, actor_ref = loading_model(args, req_input_size, get_data.output_size, get_data.max_length)
    actor_ref.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    actor_model_ref = actor_ref.to(device0)
    actor_model_ref.eval()

    obj_encoder = ObjEncoder(input_size=1, output_size=args.dim)
    obj_encoder_name = "/app/train_models/models/SFT_" + args.req_name + "_new_encoder.dict"
    obj_encoder.load_state_dict(torch.load(obj_encoder_name))
    obj_encoder = obj_encoder.to(device0)
    obj_encoder.eval()

    if args.req_name == "price":
        ppo = PPO_price(encoder, obj_encoder, actor_model, actor_model_ref, actor_optimizer, args, load_data(args))
    elif args.req_name == "bb":
        ppo = PPO_bb(encoder, obj_encoder, actor_model, actor_model_ref, actor_optimizer, args, load_data(args))
    else:
        exit()

    total_loss_train = 0
    train_len = 0

    total_loss_val = 0
    val_len = 0

    prev_val_loss = 1e6
    prev_state_dict = None

    for epoch in range(args.epoch):

        for batch_idx, (req_input, seq_output, _, new_obj) in enumerate(tqdm(train_loader)):

            loss = ppo.train(req_input, new_obj, seq_output)

            total_loss_train += loss * len(req_input)
            train_len += len(req_input)

        # with torch.no_grad():
        #     for batch_idx, (req_input, seq_output, _) in enumerate(tqdm(val_loader)):
                
        #         actor_loss, critic_loss = ppo.validate(req_input, seq_output)

        #         total_actor_loss_val += actor_loss * len(req_input)
        #         total_critic_loss_val += critic_loss * len(req_input)
        #         val_len += len(req_input)

        print(epoch, "train loss:", total_loss_train/train_len) 
        print()
        # print(epoch, "val loss:", total_actor_loss_val/val_len) 
        # print()

        decoder_checkpoint_name = "PPO_" + args.req_name + "_" + str(epoch) + "_decoder.dict"
        torch.save(actor_model.state_dict(), os.path.join(args.checkpoint_path, decoder_checkpoint_name))