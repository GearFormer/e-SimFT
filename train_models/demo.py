import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
from utils.data_handle import load_data
import os
from utils.config_file import config
from load_model import loading_model
torch.set_printoptions(threshold=10_000)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gearformer(input_, seq):
    args = config()
    max_length = 21

    get_dict = load_data(args)

    input__size = len(input_) 

    encoder, decoder = loading_model(args, input__size, get_dict.output_size, max_length)
    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint_path, args.decoder_checkpoint_name)))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_ = torch.tensor(input_).unsqueeze(0).to(torch.float32).to(device)
        encoded_input_ = encoder(input_)
        prompt = torch.zeros(len(seq))
        for i in range(len(seq)):
            prompt[i] = get_dict.name2inx(seq[i])
        prompt = prompt.to(device)

        # t = torch.tensor([[0, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52]])
        # all_logits = []
        # for i in range(20):
        #     loss_, (logits, _) = decoder(t.to(device).long(), context = encoded_input_, return_outputs = True) 
        #     t[0][i+1] = int(torch.argmax(torch.nn.functional.softmax(logits[0][i], dim = -1)))
        #     all_logits.append(logits[0][i])

        out_inx = decoder.generate(prompts=prompt.unsqueeze(0), context=encoded_input_, seq_len=21-len(seq), temperature=0)
        out = list(map(get_dict.inx2name, out_inx[0].cpu().tolist()))

        out_inx_list = out_inx[0].cpu().tolist()
        t_list = [0] + out_inx_list
        t = torch.tensor([t_list])
        loss_, (logits, _) = decoder(x=t.to(device).long(), context=encoded_input_, return_outputs=True)
    
        all_logits = logits

    return out, all_logits 


if __name__ == "__main__":
    args = config()
    get_dict = load_data(args)


    req_input = [0, 1 , 30, 0.5, 0.1, 0.2, 2, -1]  
    """
    req_input[0]: input motion type, 0 for Rotation and 1 for Translation 
    req_input[1]: output motion type, 0 for Rotation and 1 for Translation 
    req_input[2]: output speed ratio
    req_input[3], req_input[4], req_input[5]: x, y, z for output position
    req_input[6]: output motion vector direction in xyz - 0 for x, 1 for y and 2 for z
    req_input[7]: output motion vector sign: +1 for CW, -1 for CCW
    """

    current_seq = ["<start>", "translate_plus", "SH-100"]  
    
    out_seq, logits = gearformer(req_input, current_seq)
    
    print("\n\n", current_seq + out_seq, "\n\n")

    # next_token_logits = logits[0][len(current_seq)-1]
    # next_token_p = torch.nn.functional.softmax(next_token_logits, dim = -1)
    # for i in range(0, len(next_token_p)):
    #     print(get_dict.inx2name(i), next_token_p[i].item())
    # print("\n")
    # input()

    # print("Most likely next token:")
    # print(get_dict.inx2name(torch.argmax(next_token_p).item()))

    # print("\n")
