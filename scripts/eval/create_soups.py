import numpy as np
import torch
import torch.nn as nn
from esimft.utils.data_handle import DataHandler
from esimft.utils.config_file import config
from esimft.model.gearformer import GFModel, ObjEncoder
from esimft.model.gearformer_simft import GearFormerSimFT
from esimft.model.gearformer_soup import GearFormerSoup
import os


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config()

    scenarios = config.test_scenarios

    gfm = GFModel(config, device)
    encoder = gfm.encoder
    decoder = gfm.decoder
    soup_model = GearFormerSoup(config, encoder=encoder, decoder=decoder, new_req_encoder_1=ObjEncoder(input_size=1, output_size=config.dim).to(device), new_req_encoder_2=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)

    soup_model_path = os.path.join(config.checkpoint_path, "soup_models")
    os.makedirs(soup_model_path, exist_ok=True)

    for s in scenarios:
        req_list = s.split("_")
        num_test_reqs = len(req_list)

        if req_list[0] == "speed":
            model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
            model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

        elif req_list[0] == "pos":
            model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
            model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

        elif req_list[0] == "price":
            model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
            model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

        elif req_list[0] == "bb":
            model_1 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
            model_1.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

        if req_list[1] == "speed":
            model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
            model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

        elif req_list[1] == "pos":
            model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
            model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

        elif req_list[1] == "price":
            model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
            model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

        elif req_list[1] == "bb":
            model_2 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
            model_2.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))

        if num_test_reqs == 2:
            w1 = config.two_reqs_weights_1
            w2 = config.two_reqs_weights_2

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(model_1.encoder.parameters(), model_2.encoder.parameters(), soup_model.encoder.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data
                    for param1, param2, param_combined in zip(model_1.decoder.parameters(), model_2.decoder.parameters(), soup_model.decoder.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                    new_req_encoders = []
                    if req_list[0] in ["price", "bb"]:
                        new_req_encoders.append(model_1.new_req_encoder)
                    if req_list[1] in ["price", "bb"]:
                        new_req_encoders.append(model_2.new_req_encoder)
                    if len(new_req_encoders) > 0:
                        soup_model.new_req_encoder_1 = new_req_encoders[0]
                    if len(new_req_encoders) > 1:
                        soup_model.new_req_encoder_2 = new_req_encoders[1]

                soup_model_name = f"soup_{req_list[0]}_{w1[i]}_{req_list[1]}_{w2[i]}.dict"
                torch.save(soup_model.state_dict(), os.path.join(soup_model_path, soup_model_name))
                print(f"{soup_model_name} saved.")
                print()
      
        elif num_test_reqs == 3:

            if req_list[2] == "speed":
                model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.speed_model_checkpoint_name), map_location=device))

            elif req_list[2] == "pos":
                model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, device=device)
                model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.pos_model_checkpoint_name), map_location=device))

            elif req_list[2] == "price":
                model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.price_model_checkpoint_name), map_location=device))

            elif req_list[2] == "bb":
                model_3 = GearFormerSimFT(config, encoder=encoder, decoder=decoder, new_req_encoder=ObjEncoder(input_size=1, output_size=config.dim).to(device), device=device)
                model_3.load_state_dict(torch.load(os.path.join(config.checkpoint_path, config.bb_model_checkpoint_name), map_location=device))
                
            w1 = config.three_reqs_weights_1
            w2 = config.three_reqs_weights_2            
            w3 = config.three_reqs_weights_3            

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(model_1.encoder.parameters(), model_2.encoder.parameters(), soup_model.encoder.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data
                    for param1, param2, param_combined in zip(model_1.decoder.parameters(), model_2.decoder.parameters(), soup_model.decoder.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                    new_req_encoders = []
                    if req_list[1] in ["price", "bb"]:
                        new_req_encoders.append(model_2.new_req_encoder)
                    if req_list[2] in ["price", "bb"]:
                        new_req_encoders.append(model_3.new_req_encoder)
                    if len(new_req_encoders) > 0:
                        soup_model.new_req_encoder_1 = new_req_encoders[0]
                    if len(new_req_encoders) > 1:
                        soup_model.new_req_encoder_2 = new_req_encoders[1]

                soup_model_name = f"soup_{req_list[0]}_{w1[i]}_{req_list[1]}_{w2[i]}_{req_list[2]}_{w3[i]}.dict"
                torch.save(soup_model.state_dict(), os.path.join(soup_model_path, soup_model_name))   
                print(f"{soup_model_name} saved.")
                print()