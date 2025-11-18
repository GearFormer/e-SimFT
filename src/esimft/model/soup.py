import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
from train_models.utils.data_handle import load_data
from train_models.utils.config_file_soup import config
from train_models.load_model import loading_model
torch.set_printoptions(threshold=10_000)
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    args = config()
    max_length = 21
    get_dict = load_data(args)
    input_size = 8

    encoder_path = os.path.join(args.checkpoint_path, args.encoder_checkpoint_name)
    encoder_price = os.path.join(args.checkpoint_path, args.encoder_price_checkpoint_name)
    encoder_bb = os.path.join(args.checkpoint_path, args.encoder_bb_checkpoint_name)
    decoder_speed = os.path.join(args.checkpoint_path, args.decoder_speed_checkpoint_name)
    decoder_pos = os.path.join(args.checkpoint_path, args.decoder_pos_checkpoint_name)
    decoder_price = os.path.join(args.checkpoint_path, args.decoder_price_checkpoint_name)
    decoder_bb = os.path.join(args.checkpoint_path, args.decoder_bb_checkpoint_name)

    scenarios = ["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb", 
                 "speed_pos_bb", "speed_pos_price", "speed_bb_price", "pos_price_bb"]

    _, decoder1 = loading_model(args, input_size, get_dict.output_size, max_length)
    _, decoder2 = loading_model(args, input_size, get_dict.output_size, max_length)
    _, decoder3 = loading_model(args, input_size, get_dict.output_size, max_length)
    _, decoder_soup = loading_model(args, input_size, get_dict.output_size, max_length)

    for s in scenarios:

        if len(s.split("_")) == 2:
            w1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            w2 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        else:
            w1 = [0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.33]
            w2 = [0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.33]
            w3 = [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.33]

        if s == "speed_pos":
            decoder1.load_state_dict(torch.load(decoder_speed))
            decoder2.load_state_dict(torch.load(decoder_pos))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                decoder_soup_name = "soup_speed_" + str(w1[i]) + "_pos_" + str(w2[i]) + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))

        elif s == "speed_price":
            decoder1.load_state_dict(torch.load(decoder_speed))
            decoder2.load_state_dict(torch.load(decoder_price))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                decoder_soup_name = "soup_speed_" + str(w1[i]) + "_price_" + str(w2[i]) + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))

        elif s == "speed_bb":
            decoder1.load_state_dict(torch.load(decoder_speed))
            decoder2.load_state_dict(torch.load(decoder_bb))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                decoder_soup_name = "soup_speed_" + str(w1[i]) + "_bb_" + str(w2[i]) + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))

        elif s == "pos_price":
            decoder1.load_state_dict(torch.load(decoder_pos))
            decoder2.load_state_dict(torch.load(decoder_price))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                decoder_soup_name = "soup_pos_" + str(w1[i]) + "_price_" + str(w2[i]) + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))        
        
        elif s == "pos_bb":
            decoder1.load_state_dict(torch.load(decoder_pos))
            decoder2.load_state_dict(torch.load(decoder_bb))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                decoder_soup_name = "soup_pos_" + str(w1[i]) + "_bb_" + str(w2[i]) + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))        

        elif s == "price_bb":
            decoder1.load_state_dict(torch.load(decoder_price))
            decoder2.load_state_dict(torch.load(decoder_bb))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data

                decoder_soup_name = "soup_price_" + str(w1[i]) + "_bb_" + str(w2[i]) + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))       
        
        elif s == "speed_pos_bb":
            decoder1.load_state_dict(torch.load(decoder_speed))
            decoder2.load_state_dict(torch.load(decoder_pos))
            decoder3.load_state_dict(torch.load(decoder_bb))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param3, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder3.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data + w3[i] * param3.data

                decoder_soup_name = "soup_speed_" + str(w1[i]) + "_pos_" + str(w2[i]) + "_bb_" + str(w3[i])  + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))   

        elif s == "speed_pos_price":
            decoder1.load_state_dict(torch.load(decoder_speed))
            decoder2.load_state_dict(torch.load(decoder_pos))
            decoder3.load_state_dict(torch.load(decoder_price))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param3, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder3.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data + w3[i] * param3.data

                decoder_soup_name = "soup_speed_" + str(w1[i]) + "_pos_" + str(w2[i]) + "_price_" + str(w3[i])  + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))   

        elif s == "speed_bb_price":
            decoder1.load_state_dict(torch.load(decoder_speed))
            decoder2.load_state_dict(torch.load(decoder_bb))
            decoder3.load_state_dict(torch.load(decoder_price))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param3, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder3.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data + w3[i] * param3.data

                decoder_soup_name = "soup_speed_" + str(w1[i]) + "_bb_" + str(w2[i]) + "_price_" + str(w3[i])  + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))   

        elif s == "pos_price_bb":
            decoder1.load_state_dict(torch.load(decoder_pos))
            decoder2.load_state_dict(torch.load(decoder_price))
            decoder3.load_state_dict(torch.load(decoder_bb))

            for i in range(len(w1)):

                with torch.no_grad():
                    for param1, param2, param3, param_combined in zip(decoder1.parameters(), decoder2.parameters(), decoder3.parameters(), decoder_soup.parameters()):
                        param_combined.data = w1[i] * param1.data + w2[i] * param2.data + w3[i] * param3.data

                decoder_soup_name = "soup_pos_" + str(w1[i]) + "_price_" + str(w2[i]) + "_bb_" + str(w3[i])  + "_decoder.dict"
                torch.save(decoder_soup.state_dict(), os.path.join(args.checkpoint_path, decoder_soup_name))   

