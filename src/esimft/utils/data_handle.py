import random
import numpy as np
import torch
torch.manual_seed(0)
import json
import pandas as pd
from .config_file import config


class GearFormerDataset(torch.utils.data.Dataset):
    """
    x_train: if we are using weight as input (if_weight==True), it's a vector of size six [ratio(1), position(3), rotation(1), sequence_weight(1)] otherwise its [ratio(1), position(3), rotation(1)] with size five.
    y_train: it the sequence corresponding to the information in x_train
    weight: it's the sequence_weight
    """
    def __init__(self, x_train, y_train, target_length, weight):
        self.x_train = x_train.values
        self.y_train = y_train.values
        self.weight = weight.values
        self.target_length = target_length.values

    def __getitem__ (self, index):
        return torch.Tensor(self.x_train[index]).cuda(), torch.Tensor(eval(self.y_train[index])).cuda(), torch.Tensor([self.target_length[index]]).cuda(), torch.Tensor([self.weight[index]]).cuda()

    def __len__(self):
        return len(self.weight)

class PrefDataset(torch.utils.data.Dataset):
    def __init__(self, req_input, chosen_seq, reject_seq):
        self.req_input = req_input.values
        self.chosen_seq = chosen_seq.values
        self.reject_seq = reject_seq.values

    def __getitem__(self, index):
        return torch.Tensor(self.req_input[index]), torch.Tensor(self.chosen_seq[index]), \
            torch.Tensor(self.reject_seq[index])

    def __len__(self):
        return len(self.req_input)

class PrefObjDataset(torch.utils.data.Dataset):
    def __init__(self, req_input, chosen_seq, reject_seq, mid_value):
        self.req_input = req_input.values
        self.chosen_seq = chosen_seq.values
        self.reject_seq = reject_seq.values
        self.mid_value = mid_value.values

    def __getitem__(self, index):
        return torch.Tensor(self.req_input[index]), torch.Tensor(self.chosen_seq[index]), \
            torch.Tensor(self.reject_seq[index]), torch.tensor(self.mid_value[index], dtype=torch.float32)

    def __len__(self):
        return len(self.req_input)

class SMDataset(torch.utils.data.Dataset):
    def __init__(self, req_input, seq):
        self.req_input = req_input.values
        self.seq = seq.values

    def __getitem__(self, index):
        return torch.Tensor(self.req_input[index]), torch.Tensor(eval(self.seq[index]))

    def __len__(self):
        return len(self.req_input)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, req_input, seq, new_req, weights):
        self.req_input = req_input.values
        self.seq = seq.values

        if new_req is not None:
            self.new_req = new_req.values
        else:
            self.new_req = req_input.values # placeholder

        if weights is not None:
            self.weights = weights.values
        else:
            self.weights = req_input.values # placeholder

    def __getitem__(self, index):
        return torch.Tensor(self.req_input[index]), torch.Tensor(self.seq[index]), torch.Tensor(self.new_req[index]), torch.Tensor(self.weights[index])

    def __len__(self):
        return len(self.req_input)


class DataHandler:
    def __init__(self, config):
        self.config = config
        self.tokens, self.inxs = self.get_tokens()
        self.output_size = len(self.tokens)
        self.max_length = 21

    def name2inx(self, name):
        return self.tokens[name]
    
    def inx2name(self, inx):
        return self.inxs[inx]

    def json_reader(self, filename):
        with open(filename) as f:
            data=json.load(f)
        return data
    
    def get_tokens(self):
        """
        This function reads the grammar.json files and returns two dictionary:
        1. tokens: which maps the tokens we have to unique indices
        2. inxs: which maps indices to tokens
        """
        inxs = {0:"<start>"}
        tokens = {"<start>":0}
        language = self.json_reader(self.config.language_path)
        language_vocabs = language['vocab']
        cnt = 1
        for i in language_vocabs:
            for j in language_vocabs[i]:
                if j not in tokens:
                    tokens[j] = cnt
                    inxs[cnt] = j
                    cnt += 1
        tokens["<end>"] = cnt
        inxs[cnt] = "<end>"
        cnt += 1
        tokens["EOS"] = cnt
        inxs[cnt] = "EOS"
        return tokens, inxs
       
    def get_gearformer_data(self, BS, if_val, if_weight):

        """
        input:
        ------
        if_val: if True loads the val data and the output corresponds to val set, otherwise it will be train set
        if_weight: if True, weight would be used in input

        returns:
        ------
        x_train: if we are using weight as input (if_weight==True), it's a vector of size six [ratio(1), position(3), rotation(1), sequence_weight(1)] otherwise its [ratio(1), position(3), rotation(1)] with size five.
        y_train: it the sequence corresponding to the information in x_train
        weight: it's the sequence_weight
        max(weight): returns the weight of the sequence with maximum weight from the set
        """

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path

        df = pd.read_csv(folder)
        weight = df.iloc[:,0]
        target_length = df.iloc[:,1]
        x = df.iloc[:,2:-1]
        y = df.iloc[:,-1]
    
        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(GearFormerDataset(x, y, target_length, weight), batch_size= BS, shuffle=True, **kwargs)
        return loader, x.shape[1]

    # def get_all_data(self, if_val=False, if_weight=True):
    #     """
    #     input:
    #     ------
    #     if_val: if True loads the val data and the output corresponds to val set, otherwise it will be train set
    #     if_weight: if True, weight would be used in input
    #     returns:
    #     ------
    #     x_train: if we are using weight as input (if_weight==True), it's a vector of size six [ratio(1), position(3), rotation(1), sequence_weight(1)] otherwise its [ratio(1), position(3), rotation(1)] with size five.
    #     y_train: it the sequence corresponding to the information in x_train
    #     weight: it's the sequence_weight
    #     max(weight): returns the weight of the sequence with maximum weight from the set
    #     """

    #     if if_val:
    #         folder = "/app/simulator/test_data.csv"
    #     else:
    #         folder = self.config.train_data_path

    #     df = pd.read_csv(folder)
    #     weight = df.iloc[:,0]
    #     target_length = df.iloc[:,1]
    #     x_train = df.iloc[:,2:-1]
    #     y_train = df.iloc[:,-1]

    #     return x_train, y_train, target_length, weight, 0
    
    def get_sft_data(self, BS, if_val):

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path
        
        df = pd.read_pickle(folder)

        req_input = df.iloc[:,0]
        chosen_seq = df.iloc[:,1]
        obj_list = None
        weights = None

        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(SFTDataset(req_input, chosen_seq, obj_list, weights), batch_size=BS, shuffle=True, **kwargs)
        
        return loader

    def get_sft_obj_data(self, BS, if_val):

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path
        
        df = pd.read_pickle(folder)

        req_input = df.iloc[:,0]
        chosen_seq = df.iloc[:,1]
        obj_list = df.iloc[:,2]
        weights = None

        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(SFTDataset(req_input, chosen_seq, obj_list, weights), batch_size=BS, shuffle=True, **kwargs)
        
        return loader

    def get_sft_ric_data(self, BS, if_val):

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path
        
        df = pd.read_pickle(folder)

        req_input = df.iloc[:,0]
        seq = df.iloc[:,1]
        new_req = df.iloc[:,2]
        weights = df.iloc[:,3]

        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(SFTDataset(req_input, seq, new_req, weights), batch_size=BS, shuffle=True, **kwargs)
        
        return loader
    
    def get_pref_data(self, BS, if_val):

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path
        
        df = pd.read_pickle(folder)

        req_input = df.iloc[:,0]
        chosen_seq = df.iloc[:,1]
        reject_seq = df.iloc[:,2]

        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(PrefDataset(req_input, chosen_seq, reject_seq), batch_size=BS, shuffle=True, **kwargs)
        
        return loader
    
    def get_pref_obj_data(self, BS, if_val):

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path
        
        df = pd.read_pickle(folder)

        req_input = df.iloc[:,0]
        chosen_seq = df.iloc[:,1]
        reject_seq = df.iloc[:,2]
        mid_value = df.iloc[:,3]

        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(PrefObjDataset(req_input, chosen_seq, reject_seq, mid_value), batch_size=BS, shuffle=True, **kwargs)
        
        return loader

    def get_sm_data(self, BS, if_val):

        if if_val:
            folder = self.config.val_data_path
        else:
            folder = self.config.train_data_path
        
        df = pd.read_csv(folder)

        req_input = df.iloc[:,2:-1]
        seq = df.iloc[:,-1]

        kwargs = {'num_workers': 0} if torch.cuda.is_available() else {}
        loader = torch.utils.data.DataLoader(SMDataset(req_input, seq), batch_size=BS, shuffle=True, **kwargs)
        
        return loader       

if __name__ == "__main__":

 from config_file import config
 config = config()
 ld = load_data(config)
 print(ld.tokens)
 print(ld.inxs)