import torch
import torch.nn as nn
import torch.optim as optim


class GearFormerSimFT(nn.Module):

    def __init__(
        self, config,
        encoder, decoder, new_req_encoder=None, weight_encoder=None, 
        freeze_encoder=True, freeze_new_req_encoder=True, freeze_weight_encoder=True, ric=False, device="cuda"):

        super(GearFormerSimFT, self).__init__()

        self.freeze_encoder = freeze_encoder
        self.freeze_new_req_encoder = freeze_new_req_encoder
        self.freeze_weight_encoder = freeze_weight_encoder
        self.ric = ric
        self.device = device
        
        self.encoder = encoder
        if self.freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()
            
        self.decoder = decoder
        self.decoder.train()
        optimizer_params = [{"params": self.decoder.parameters(), "lr": config.lr}]

        self.new_req_encoder = new_req_encoder
        if self.new_req_encoder is not None:
            if self.freeze_new_req_encoder:
                self.new_req_encoder.eval()
            else:
                self.new_req_encoder.train()
                optimizer_params.append({"params": self.new_req_encoder.parameters(), "lr": config.lr})

        self.weight_encoder = weight_encoder
        if self.weight_encoder is not None:
            if self.freeze_weight_encoder:
                self.weight_encoder.eval()
            else:
                self.weight_encoder.train()
                optimizer_params.append({"params": self.weight_encoder.parameters(), "lr": config.lr})

        self.optimizer = optim.Adam(optimizer_params)

        self.gen_seq_len = config.max_length - 2

    def forward(self, inputs, labels):

        req_input = inputs[0].to(self.device)
        encoded_input = self.encoder(req_input)

        if self.new_req_encoder is not None:
            if self.ric:
                new_req_list = inputs[1][:, :2].to(self.device)
            else:   
                new_req_list = inputs[1][:,0].unsqueeze(-1).to(self.device)
            encoded_new_req = self.new_req_encoder(new_req_list)
            encoded_input += encoded_new_req.unsqueeze(1)

        if self.weight_encoder is not None:
            weights = inputs[2].to(self.device)
            encoded_weights = self.weight_encoder(weights)
            encoded_input += encoded_weights.unsqueeze(1)

        labels = labels.long().to(self.device)

        _, (logits, _) = self.decoder(labels, context = encoded_input, return_outputs = True) 

        return logits

    def generate(self, inputs, prompts, temperature=1.0):

        req_input = inputs[0].to(self.device)
        encoded_input = self.encoder(req_input)

        if self.new_req_encoder is not None:
            if self.ric:
                new_req_list = torch.cat([inputs[1], inputs[2]], dim=1).to(self.device)
            else:
                new_req_list = inputs[1][:,0].unsqueeze(-1).to(self.device)
            encoded_new_req = self.new_req_encoder(new_req_list)
            encoded_input += encoded_new_req.unsqueeze(1)

        if self.weight_encoder is not None:
            weights = inputs[3].to(self.device) if self.ric else inputs[2].to(self.device)
            encoded_weights = self.weight_encoder(weights)
            encoded_input += encoded_weights.unsqueeze(1)

        sampled = self.decoder.generate(prompts=prompts, context=encoded_input, seq_len=self.gen_seq_len, temperature=temperature)

        return sampled