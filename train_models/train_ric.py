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
from .sft_ric import train_sft, val_sft
from .transformers import WeightEncoder, ObjEncoder
from tqdm import tqdm


if __name__ == "__main__":
    
    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = config()    
    model = args.model_name
    get_data = load_data(args)
    req_input_size = 8

    train_loader = get_data.get_sft_ric_data(args.BS, if_val=False)
    val_loader = get_data.get_sft_ric_data(args.BS, if_val=True)
    
    encoder, decoder = loading_model(args, req_input_size, get_data.output_size, get_data.max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    encoder = encoder.to(device0)
    encoder.eval()
    decoder = decoder.to(device0)

    obj_encoder = ObjEncoder(input_size=2, output_size=args.dim)
    obj_encoder = obj_encoder.to(device0)
    w_encoder = WeightEncoder(input_size=4, output_size=args.dim)
    w_encoder = w_encoder.to(device0)

    obj_encoder_optimizer = optim.Adam(obj_encoder.parameters(), lr=1e-6)
    w_encoder_optimizer = optim.Adam(w_encoder.parameters(), lr=1e-6)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-6)

    prev_val_loss = 1e6
    decoder_state_dict = None
    encoder_state_dict = None

    for epoch in range(args.epoch):

        total_loss_val, val_len = 0, 0
        total_loss_train, train_len = 0, 0
            
        for batch_idx, (req_input, seq, new_req, weights) in enumerate(tqdm(train_loader)):

            req_input = req_input.to(device0)
            seq = seq.to(device0)
            new_req = new_req.to(device0)
            weights = weights.to(device0)

            obj_encoder.train()
            w_encoder.train()

            encoded_input = encoder(req_input)
            encoded_new_req = obj_encoder(new_req)
            encoded_w = w_encoder(weights)

            input_data = encoded_input + encoded_new_req + encoded_w

            loss = train_sft(input_data, seq, decoder, decoder_optimizer, obj_encoder_optimizer, w_encoder_optimizer)
            total_loss_train += loss * len(req_input)
            train_len += len(req_input)

        for batch_idx, (req_input, seq, new_req, weights) in enumerate(tqdm(val_loader)):

            req_input = req_input.to(device0)
            seq = seq.to(device0)
            new_req = new_req.to(device0)
            weights = weights.to(device0)

            obj_encoder.eval()
            w_encoder.eval()

            encoded_input = encoder(req_input)
            encoded_new_req = obj_encoder(new_req)
            encoded_w = w_encoder(weights)

            input_data = encoded_input + encoded_new_req + encoded_w

            loss = val_sft(input_data, seq, decoder)
            total_loss_val += loss * len(req_input)
            val_len += len(req_input)

        print(epoch, "train Loss:", total_loss_train/train_len) 
        print(epoch, "val Loss:", total_loss_val/val_len)

        if total_loss_val < prev_val_loss:
            obj_encoder_state_dict = obj_encoder.state_dict()
            w_encoder_state_dict = w_encoder.state_dict()
            decoder_state_dict = decoder.state_dict()
            prev_val_loss = total_loss_val
        else:
            obj_encoder_checkpoint_name = "SFT_ric_obj_encoder.dict"
            w_encoder_checkpoint_name = "SFT_ric_w_encoder.dict"
            decoder_checkpoint_name = "SFT_ric_decoder.dict"
            torch.save(obj_encoder_state_dict, os.path.join(args.checkpoint_path, obj_encoder_checkpoint_name))
            torch.save(w_encoder_state_dict, os.path.join(args.checkpoint_path, w_encoder_checkpoint_name))
            torch.save(decoder_state_dict, os.path.join(args.checkpoint_path, decoder_checkpoint_name))
            exit()

        # if epoch == args.epoch-1:
        #     encoder_state_dict = obj_encoder.state_dict()
        #     decoder_state_dict = decoder.state_dict()
        #     encoder_checkpoint_name = "SFT_bb_encoder.dict"
        #     decoder_checkpoint_name = "SFT_bb_decoder.dict"
        #     torch.save(encoder_state_dict, os.path.join(args.checkpoint_path, encoder_checkpoint_name))
        #     torch.save(decoder_state_dict, os.path.join(args.checkpoint_path, decoder_checkpoint_name))

