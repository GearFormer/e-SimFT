import torch
import numpy as np
torch.manual_seed(0)
from esimft.model.functions import compute_cross_entropy_loss
from esimft.model.gearformer import ObjEncoder, WeightEncoder
import torch.optim as optim


class SFT:
    def __init__(self, config, model, freeze_encoder=True, fit_new_req=False, fit_req_weights=False):

        self.config = config
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.freeze_encoder = freeze_encoder
        self.fit_new_req = fit_new_req
        self.fit_req_weights = fit_req_weights

        if freeze_encoder:
            self.encoder.eval()
        else:
            self.encoder.train()
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config.lr)

        self.decoder.train()
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=config.lr)

        if fit_new_req:
            if fit_req_weights:
                self.new_req_encoder = ObjEncoder(input_size=2, output_size=config.dim)
                self.weight_encoder = WeightEncoder(input_size=4, output_size=config.dim)
                self.weight_encoder.to(model.device)

                self.weight_encoder_optimizer = optim.Adam(self.weight_encoder.parameters(), lr=config.lr)

            else:
                self.new_req_encoder = ObjEncoder(input_size=1, output_size=config.dim)
            
            self.new_req_encoder.to(model.device)
            self.new_req_encoder_optimizer = optim.Adam(self.new_req_encoder.parameters(), lr=config.lr)


    def training_step(self, input_data, target, ignore_index=52, new_req_list=None, weights=None):
        
        if not self.freeze_encoder:
            self.encoder_optimizer.zero_grad()

        self.decoder_optimizer.zero_grad()
        
        encoded_input = self.encoder(input_data)

        if self.fit_new_req and new_req_list is not None:
            self.new_req_encoder_optimizer.zero_grad()
            encoded_new_req = self.new_req_encoder(new_req_list)
            encoded_input += encoded_new_req

        if self.fit_req_weights and weights is not None:
            self.weight_encoder_optimizer.zero_grad()
            encoded_weights = self.weight_encoder(weights)
            encoded_input += encoded_weights

        target = target.long()

        _, (logits, _) = self.decoder(target, context = encoded_input, return_outputs = True) 
        
        loss = compute_cross_entropy_loss(logits, target[:, 1:], ignore_index=ignore_index)

        loss.backward()

        if not self.freeze_encoder:
            self.encoder_optimizer.step()

        if self.fit_new_req:
            self.encoded_new_req.step()

        if self.fit_req_weights:
            self.weight_encoder_optimizer.step()

        self.decoder_optimizer.step()

        return loss.item()

    def validation_step(self, input_data, target, ignore_index=52, new_req_list=None, weights=None):
        
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            encoded_input = self.encoder(input_data)

            if self.fit_new_req and new_req_list is not None:
                self.new_req_encoder.eval()
                encoded_new_req = self.new_req_encoder(new_req_list)
                encoded_input += encoded_new_req

            if self.fit_req_weights and weights is not None:
                self.weight_encoder.eval()
                encoded_weights = self.weight_encoder(weights)
                encoded_input += encoded_weights

            target = target.long()

            _, (logits, _) = self.decoder(target, context = encoded_input, return_outputs = True) 
            
            loss = compute_cross_entropy_loss(logits, target[:, 1:], ignore_index=ignore_index)

        return loss.item()