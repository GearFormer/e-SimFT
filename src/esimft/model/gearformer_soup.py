import torch
import torch.nn as nn
import torch.optim as optim


class GearFormerSoup(nn.Module):

    def __init__(
        self, config,
        encoder, decoder, new_req_encoder_1=None, new_req_encoder_2=None, device="cuda"):

        super(GearFormerSoup, self).__init__()

        self.device = device
        
        self.encoder = encoder
        self.encoder.eval()
            
        self.decoder = decoder
        self.decoder.eval()

        self.new_req_encoder_1 = new_req_encoder_1
        self.new_req_encoder_1.eval()

        self.new_req_encoder_2 = new_req_encoder_2
        self.new_req_encoder_2.eval()

        self.gen_seq_len = config.max_length - 2

    def forward(self, inputs, labels):

        req_input = inputs[0].to(self.device)
        encoded_input = self.encoder(req_input)

        if self.new_req_encoder_1 is not None:
            new_req_list = inputs[1][:,0].unsqueeze(-1).to(self.device)
            encoded_new_req = self.new_req_encoder_1(new_req_list)
            encoded_input += encoded_new_req.unsqueeze(1)

        if self.new_req_encoder_2 is not None:
            new_req_list = inputs[2][:,0].unsqueeze(-1).to(self.device)
            encoded_new_req = self.new_req_encoder_2(new_req_list)
            encoded_input += encoded_new_req.unsqueeze(1)

        labels = labels.long().to(self.device)

        _, (logits, _) = self.decoder(labels, context = encoded_input, return_outputs = True) 

        return logits

    def generate(self, inputs, prompts, temperature=1.0):

        req_input = inputs[0].to(self.device)
        encoded_input = self.encoder(req_input)

        if self.new_req_encoder_1 is not None:
            new_req_list = inputs[1][:,0].unsqueeze(-1).to(self.device)
            encoded_new_req = self.new_req_encoder_1(new_req_list)
            encoded_input += encoded_new_req.unsqueeze(1)

        if self.new_req_encoder_2 is not None:
            new_req_list = inputs[2][:,0].unsqueeze(-1).to(self.device)
            encoded_new_req = self.new_req_encoder_2(new_req_list)
            encoded_input += encoded_new_req.unsqueeze(1)

        sampled = self.decoder.generate(prompts=prompts, context=encoded_input, seq_len=self.gen_seq_len, temperature=temperature)

        return sampled