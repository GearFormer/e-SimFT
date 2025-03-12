import json 
import argparse
import numpy as np
np.random.seed(100)
import random
random.seed(100)
import os
from decimal import Decimal as D
from decimal import getcontext
import time
import math
import pickle
getcontext().prec = 5



def json_writer(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f)

def json_reader(filename):
    with open(filename) as f:
        data=json.load(f)
    return data

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_path', type=str, default='language.json', help='path to the language.json')
    parser.add_argument('--max_length', type=int, default=10, help='maximum number of components - >= 2')
    parser.add_argument('--catalogue_path', type=str, default='catalogue.json', help='path to the catalogue.json')
    args = parser.parse_args()
    return args

def save_sequences_json(args, generated_seqs):
    if not os.path.exists(os.path.join("examples", str(args.max_length))):
        os.makedirs(os.path.join("examples", str(args.max_length)))
    cnt = 0
    for i in generated_seqs:
        cnt += 1
        input_rot_axis = random.choice([1,-1])
        input_speed = random.uniform(1, 628)

        data = {
            "input_params": {
            "input_position": [0, 0, 0],
            "input_rot_axis": [0, input_rot_axis, 0],
            "input_speed": round(input_speed,2)
            },
            "gear_train_sequence": list(i)
        }
        
        file_name = "example_MaxComponent_"+str(args.max_length)+"_"+str(cnt)+".json"
        file_path = os.path.join("examples", str(args.max_length), file_name)
        json_writer(data, file_path)

class generate_sequences:

    def __init__(self, args):
        self.args = args
        if self.args.max_length < 2:
            print("The maximum number of components should be at least 2")
            return 0


    def get_number_of_components(self, sequence):
        """
        Given a sequence computes how many components it has, 
        IMPORTANT: because we added start and end to the number of componenets we 
        only remove mesh and Translate to compute the number of components.
        """
        # print(sequence)
        elements_to_remove = {"Mesh", "Translate", "<start>", "<end>"}
        return len([word for word in sequence if word not in elements_to_remove])

    def get_policy(self):
        self.catalogue = json_reader(self.args.catalogue_path)
        language = json_reader(self.args.language_path)
        grammar_list = language['grammar']
        self.grammar = {}
        for elem in grammar_list:
            # self.grammar[tuple(elem['LHS'])] = [tuple(sublist) for sublist in elem['RHS']] ### for perious version of grammar
            self.grammar[elem['LHS']] = [sublist for sublist in elem['RHS']]
        self.vocab = language['vocab']

    def add_to_seq(self, current_sequence):
        """
        Given an incoplete sequence format of length L,
        it returns all the valid sequence formats of length L+1
        """
        possible_seqs = []    
        # print(current_sequence)
        for rhs in self.grammar[current_sequence[-1]]:
            # print(rhs)
            # print(current_sequence)
            possible_seqs.append(current_sequence + rhs)
        return possible_seqs

    def generate_seq_format(self):
        """
        Generates all the possible sequence formats
        using the grammer in the language.json
        """
        self.get_policy()
        
        completed_seq = set()
        processing_seqs = [["<start>"]]

        while len(processing_seqs) > 0 :
            # print(processing_seqs)
            for elem in processing_seqs:
                if elem[-1] == "<end>":
                    if self.get_number_of_components(elem) > 1 and (self.get_number_of_components(elem) < self.args.max_length + 1):
                        completed_seq.add(tuple(elem))
                else:
                    if self.get_number_of_components(elem) < self.args.max_length + 1:
                        processing_seqs += self.add_to_seq(elem)
                        
                processing_seqs.remove(elem)

        return completed_seq
    
    def get_seqs_by_format(self, seq_format):
        """
        Given one specific sequence format, it outputs all the possible
        sequences which map to that format
        """

        flattened_format = seq_format
        
        all_seq_per_format = [('<start>',)]
        for elem in flattened_format[1:]:
            if elem not in self.vocab:   ### this is for <start> and <end>
                for i in range(len(all_seq_per_format)):
                    all_seq_per_format[i] += (elem,)
            else:
                all_seq_base = list(all_seq_per_format)
                all_seq_per_format = []

                voc = self.vocab[elem]
                for i in voc:
                    for j in range(len(all_seq_base)):
                        all_seq_per_format.append(all_seq_base[j]+ (i,))
        return set(all_seq_per_format)



    def generate_all_seq_by_format_and_check_feasibility(self, seq_format):
        """
        Generates all the feasible sequences
        """
        self.get_policy()
        feasible_seq_firmat = set()
        all_seq_by_format = self.get_seqs_by_format(seq_format)
        for seq in all_seq_by_format:
            if feasible.is_physically_feasible(seq, self.catalogue):
                feasible_seq_firmat.add(seq)
            else:
                pass  
        return feasible_seq_firmat

    def generate_sequence_random(self, seq_format, format_number):
        self.get_policy()
        num_of_comps = self.get_number_of_components(seq_format)
        if num_of_comps > 5:

            num_of_seq = 2**(num_of_comps)

            num_of_seq = min(num_of_seq, 200)
        else:
            seqs = self.generate_all_seq_by_format_and_check_feasibility(seq_format)
            num_of_seq = len(seqs)//2
            num_of_seq = min(num_of_seq, 200)
            
        seq_format = seq_format[1:-1]

        with open("examples/format_number_"+str(format_number)+"_length_"+str(num_of_comps)+".pickle", 'wb') as csvfile:
            
            pickler = pickle.Pickler(csvfile)
            all_seq = set()
            while (num_of_seq > 0):
                seq = []
                for i in seq_format:
                    seq.append(random.choice(self.vocab[i]))
                if feasible.is_physically_feasible(['<start>'] + seq + ['<end>'], self.catalogue):
                    if str(seq) not in all_seq:
                        all_seq.add(str(seq))
                        pickler.dump(seq)
                        num_of_seq -= 1



class feasible:
    @staticmethod
    def is_physically_feasible(seq, catalogue):
        """
        Given a sequence, we want to see if 
        it is physically feasible or not
        """
        hypoid_sign = 1
        occupied = []
        ortho1 = {"x":"y", "y":"z", "z":"x"}
        ortho2 = {"x":"z", "y":"x", "z":"y"}
        dir_to_vec = {"x":np.array([1,0,0]), "y":np.array([0,1,0]), "z":np.array([0,0,1])}
        needed_space = 0
        curr = np.array([D(0),D(0),D(0)])
        i = 0
        curr_dir = "y"

        mhpl_translate = False

        while (i < len(seq)-1):
            i += 1          
            if "MSGA" in seq[i] or "SBSG" in seq[i] or "MMSG" in seq[i] or "AG" in seq[i] or "SWG" in seq[i] or ("MHP" in seq[i] and "R" in seq[i]):
                radius = D(catalogue[seq[i]]["outside_dia"] / 2)
                width = D(catalogue[seq[i]]["total_length"] / 2)
                bb = dir_to_vec[curr_dir]*width + dir_to_vec[ortho1[curr_dir]]*radius + dir_to_vec[ortho2[curr_dir]]*radius
                needed_space = ((curr[0]-bb[0], curr[0]+bb[0]), (curr[1]-bb[1], curr[1]+bb[1]), (curr[2]-bb[2], curr[2]+bb[2]))
                
            elif "MRGF" in seq[i]:
                width = D(catalogue[seq[i]]["face_width"]/2)
                height = D(catalogue[seq[i]]["height"] / 2)
                length = D(catalogue[seq[i]]["total_length"])
                if seq[i-1] == "<start>":
                    curr_dir = "x"
                    if "1st" in seq[i+1]:
                        pitch_side = "y"
                        width_side = "z"
                    else:
                        pitch_side = "z"
                        width_side = "y"
                    bb = dir_to_vec[curr_dir]*length + dir_to_vec[pitch_side]*height + dir_to_vec[width_side]*width
                    needed_space = ((curr[0], curr[0]+2*length), (curr[1]-bb[1], curr[1]+bb[1]), (curr[2]-bb[2], curr[2]+bb[2]))

                elif seq[i+1] == "<end>":
                    a = ["x", "y", "z"]
                    a.remove(curr_dir)
                    a.remove(pitch_side)
                    if len(a) != 1:
                        Error
                    else:
                        width_side = a[0]
                    
                    bb = dir_to_vec[curr_dir]*length + dir_to_vec[pitch_side]*height + dir_to_vec[width_side]*width
                    needed_space = ((curr[0]-bb[0], curr[0]+bb[0]), (curr[1]-bb[1], curr[1]+bb[1]), (curr[2]-bb[2], curr[2]+bb[2]))


            elif "MHP" in seq[i] and "L" in seq[i]:
                length = D(catalogue[seq[i]]["total_length"])
                radius = D(catalogue[seq[i]]["shaft_dia"] / 2)
                if curr_dir == "x":
                    if hypoid_sign == 1:
                        needed_space = ((curr[0], curr[0]+length), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))
                    else:
                        needed_space = ((curr[0]-length, curr[0]), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))

                if curr_dir == "y":
                    if hypoid_sign == 1:
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1], curr[1]+length), (curr[2]-radius, curr[2]+radius))
                    else:
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-length, curr[1]), (curr[2]-radius, curr[2]+radius))

                if curr_dir == "z":
                    if hypoid_sign == 1:
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2], curr[2]+length))
                    else:
                        needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2]-length, curr[2]))      


            elif "mesh" in seq[i]:
                if "plus" in seq[i]: sign_ = 1
                else: sign_ = -1
                
                if "MSGA" in seq[i+1] and "MSGA" in seq[i-1]:   ### spur
                    r1 = D(catalogue[seq[i-1]]["pitch_dia"])/2
                    r2 = D(catalogue[seq[i+1]]["pitch_dia"])/2

                    if "1st" in seq[i]: 
                        curr += dir_to_vec[ortho1[curr_dir]] * sign_ * (r1+r2)

                    else:
                        curr += dir_to_vec[ortho2[curr_dir]] * sign_ * (r1+r2)



                elif "MHP" in seq[i+1] and "MHP" in seq[i-1]:   ### bevel hypoid and pinion
                    if "R" in seq[i-1]:   ### i+1 is hypoid and i-1 is bevel
                        hypoid_sign *= -1
                        e1 = D(catalogue[seq[i-1]]["mounting_distance"])
                        i1 = D(catalogue[seq[i-1]]["total_length"])/2
                        e2 = D(catalogue[seq[i+1]]["mounting_distance"])
                        l = D(catalogue[seq[i-1]]['offset'])

                        if "1st" in seq[i]:
                            curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho1[curr_dir]]*(sign_*l) + dir_to_vec[ortho2[curr_dir]]*(sign_*e2)
                            curr_dir = ortho2[curr_dir]

                        elif "2nd" in seq[i]:
                            curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho2[curr_dir]]*(sign_*l) + dir_to_vec[ortho1[curr_dir]]*(-1*sign_*e2)
                            curr_dir = ortho1[curr_dir]

                    
                    if "L" in seq[i-1]: ### i+1 is bevel and i-1 is hypoid
                        e1 = D(catalogue[seq[i-1]]["mounting_distance"])
                        i2 = D(catalogue[seq[i+1]]["total_length"])/2
                        e2 = D(catalogue[seq[i+1]]["mounting_distance"])
                        l = D(catalogue[seq[i-1]]['offset'])

                        if "1st" in seq[i]:
                            curr += dir_to_vec[curr_dir]*e1*t_sign + dir_to_vec[ortho1[curr_dir]]*(-1*sign_*l) + dir_to_vec[ortho2[curr_dir]]*(-1*sign_*(e2-i2))
                            curr_dir = ortho2[curr_dir]
                        
                        elif "2nd" in seq[i]:
                            curr += dir_to_vec[curr_dir]*e1*t_sign + dir_to_vec[ortho1[curr_dir]]*(-1*sign_*(e2-i2)) + dir_to_vec[ortho2[curr_dir]]*(sign_*l)
                            curr_dir = ortho1[curr_dir]
                        

                elif ("SBSG" in seq[i+1] and "SBSG" in seq[i-1]) or ("MMSG" in seq[i+1] and "MMSG" in seq[i-1]):   ### bevel or meter
                        
                    e1 = D(catalogue[seq[i-1]]["mounting_distance"])
                    e2 = D(catalogue[seq[i+1]]["mounting_distance"]) 
                    i1 = D(catalogue[seq[i-1]]["total_length"]/2)
                    i2 = D(catalogue[seq[i+1]]["total_length"]/2)
                    
                    if "1st" in seq[i]:
                        curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho1[curr_dir]]*sign_*(e2-i2)
                        curr_dir = ortho1[curr_dir]

                    if "2nd" in seq[i]:
                        curr += dir_to_vec[curr_dir]*(e1-i1)*t_sign + dir_to_vec[ortho2[curr_dir]]*sign_*(e1-i1)
                        curr_dir = ortho2[curr_dir]
                        

                elif "SWG" in seq[i-1] or "SWG" in seq[i+1]:
                    r1 = D(catalogue[seq[i-1]]["pitch_dia"])/2
                    r2 = D(catalogue[seq[i+1]]["pitch_dia"])/2
                    if "1st" in seq[i]:
                        curr += dir_to_vec[ortho1[curr_dir]]*sign_*(r1+r2)
                        curr_dir = ortho2[curr_dir]

                    elif "2nd" in seq[i]:
                        curr += dir_to_vec[ortho2[curr_dir]]*sign_*(r1+r2)
                        curr_dir = ortho1[curr_dir]


                elif "MRGF" in seq[i-1] or "MRGF" in seq[i+1]:
                    if "MRGF" in seq[i-1]:
                        total_length = D(catalogue[seq[i-1]]["total_length"])
                        height = D(catalogue[seq[i-1]]["height_to_pitch"])/2    ### chose height(high_to_pich_line) for rack and pitch_dia(out_side_dia) for gear
                        pitch_dia = D(catalogue[seq[i+1]]["pitch_dia"])/2
                        if "1st" in seq[i]:
                            curr += dir_to_vec[curr_dir]*total_length + dir_to_vec[ortho1[curr_dir]]*sign_*(height+pitch_dia)
                            curr_dir = ortho2[curr_dir]

                        elif "2nd" in seq[i]:
                            curr += dir_to_vec[curr_dir]*total_length + dir_to_vec[ortho2[curr_dir]]*sign_*(height+pitch_dia)
                            curr_dir = ortho1[curr_dir]

                    elif "MRGF" in seq[i+1]: ### this is the end we do not need the curr anymore
                        total_length = D(catalogue[seq[i+1]]["total_length"])
                        height = D(catalogue[seq[i+1]]["height_to_pitch"])/2    ### chose height(high_to_pich_line) for rack and pitch_dia(out_side_dia) for gear
                        pitch_dia = D(catalogue[seq[i-1]]["pitch_dia"])/2
                        if "1st" in seq[i]:
                            curr += dir_to_vec[ortho1[curr_dir]]*sign_*(height+pitch_dia)
                            curr_dir = ortho2[curr_dir]
                            pitch_side = ortho1[curr_dir]

                        elif "2nd" in seq[i]:
                            curr += dir_to_vec[ortho2[curr_dir]]*sign_*(height+pitch_dia)
                            curr_dir = ortho1[curr_dir]
                            pitch_side = ortho2[curr_dir]
               

            elif "translate" in seq[i]:
                if ("MHP" in seq[i-1]) and ("L" in seq[i-1]):
                    mhpl_translate = True

                if "plus" in seq[i]: t_sign = +1
                else: t_sign = -1

                if seq[i+1] == "SH-*":
                    length = 0
                    if seq[i-1] == "<start>" or ("MHP" in seq[i-1] and "R" in seq[i-1]):
                        pass
                    else:                        
                        length += D(catalogue[seq[i-1]]["total_length"] / 2)
                    
                    if seq[i+2] == "<end>" or ("MHP" in seq[i-1] and "R" in seq[i-1]):
                        pass
                    else:
                        length += D(catalogue[seq[i+2]]["total_length"] / 2)

                    
                    if curr_dir == "y":
                        curr = (curr[0], curr[1]+t_sign*length, curr[2])
                    if curr_dir == "x":
                        curr = (curr[0]+t_sign*length, curr[1], curr[2])
                    if curr_dir == "z":
                        curr = (curr[0], curr[1], curr[2]+t_sign*length)


                else:
                    length = D(catalogue[seq[i+1]]["total_length"])
                    radius = D(catalogue[seq[i+1]]["outside_dia"]/2)

                    if "plus" in seq[i]:
                        hypoid_sign = +1
                        if curr_dir == "y":
                            needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1], curr[1]+length), (curr[2]-radius, curr[2]+radius))
                            curr = (curr[0], curr[1]+length, curr[2])
                        if curr_dir == "x":
                            needed_space = ((curr[0], curr[0]+length), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))
                            curr = (curr[0]+length, curr[1], curr[2])
                        if curr_dir == "z":
                            needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2], curr[2]+length))
                            curr = (curr[0], curr[1], curr[2]+length)
                    else:
                        hypoid_sign = -1
                        if curr_dir == "y":
                            needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-length, curr[1]), (curr[2]-radius, curr[2]+radius))
                            curr = (curr[0], curr[1]-length, curr[2])
                        if curr_dir == "x":
                            needed_space = ((curr[0]-length, curr[0]), (curr[1]-radius, curr[1]+radius), (curr[2]-radius, curr[2]+radius))
                            curr = (curr[0]-length, curr[1], curr[2])
                        if curr_dir == "z":
                            needed_space = ((curr[0]-radius, curr[0]+radius), (curr[1]-radius, curr[1]+radius), (curr[2]-length, curr[2]))
                            curr = (curr[0], curr[1], curr[2]-length)

                    
                    i += 1
            if needed_space != 0:
                if feasible.possible(occupied, needed_space, mhpl_translate):
                    occupied.append(needed_space)
                    mhpl_translate = False
                    needed_space = 0
                else:
                    return False
        return True            
   


    @staticmethod
    def possible(occupied, needed_space, mhpl_translate):
        if not mhpl_translate:
            occupied = occupied[:-1]  
        for i in occupied:
            if feasible.intervals_collide(i[1], needed_space[1]):
                if feasible.intervals_collide(i[0], needed_space[0]):
                    if feasible.intervals_collide(i[2], needed_space[2]):
                        return False
        return True
            
    @staticmethod
    def intervals_collide(interval1, interval2):
        if interval2[0] == interval1[0] and interval2[1] == interval1[1]:
            return True
        for point in interval1:
            if (interval2[0] < point < interval2[1]):
                return True
        for point in interval2:
            if (interval1[0] < point < interval1[1]):
                return True
        return False
    
