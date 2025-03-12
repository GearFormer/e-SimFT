import json 
import argparse
import random
import os
from decimal import Decimal as D
from decimal import getcontext
import time
getcontext().prec = 5
from generate_feasible_sequences import json_writer, generate_sequences, json_reader
import sys

def save_sequences_json(args, generated_seqs, seq_number):
    if not os.path.exists(os.path.join("examples", str(args.max_length - 2))):
        os.makedirs(os.path.join("examples", str(args.max_length - 2)))
    cnt = 0
    for i in generated_seqs:
        cnt += 1
        data = {
            "gear_train_sequence": list(i)
        }
        
        file_name = "S"+str(seq_number)+"_"+str(cnt)+".json"
        file_path = os.path.join("examples", str(args.max_length-2), file_name)
        json_writer(data, file_path)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_path', type=str, default='language.json', help='path to the language.json')
    parser.add_argument('--max_length', type=int, default=10, help='maximum number of components >= 2')
    parser.add_argument('--catalogue_path', type=str, default="catalogue.json", help='path to the catalogue.json')
    parser.add_argument('--num_of_cpus', type=int, default=4, help="how many cpus we want to use")
    parser.add_argument('--cpu_num', type=int, default=0, help="which cpu is")
    parser.add_argument('--random_sample_vocab', type=bool, default=True, help='if true instead of all the vocab list randomly select number_sample_vocab')
    parser.add_argument('--number_sample_vocab', type=int, default=2, help = "random samples from vocab's lists")
    parser.add_argument('--generate_json_for_format', type=bool, default=False, help = "random samples from vocab's lists")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = config()    
    seq_class = generate_sequences(args)

    
    #to generate format_<max_length>.json you can set generate_json_for_format to true. Currently max_length is 10 and we have format_10.json, but after changing the max_length we have to regenerate the corresponding .json file.
    if args.generate_json_for_format: 
        generated_format = seq_class.generate_seq_format()
        data = {}
        data["formats"] = list(generated_format)
        json_writer(data, "format_"+ str(args.max_length) +".json")
        sys.exit()    

    
    formats_json = json_reader("format_"+ str(args.max_length) +".json")
    generated_format = list(formats_json["formats"])


    cnt = args.cpu_num
    while(cnt < len(generated_format)):
        print("start:  ", cnt)
        seq_format = generated_format[cnt]
        seqs_for_format = seq_class.generate_sequence_random(seq_format, cnt)
        print("done: ", cnt)
        cnt += args.num_of_cpus
