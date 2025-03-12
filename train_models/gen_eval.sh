#!/bin/bash

for i in {8..8}
do
    python evaluation.py --model_name ab_SFT_pos_"$i" --checkpoint_path /app/gearformer_model/models/ --encoder_checkpoint_name Xtransformer_0.0001_18_encoder.dict --decoder_checkpoint_name ab_SFT_pos_"$i"_decoder.dict --val_data_path /app/simulator/test_data.csv --BS 1024
    # CUDA_VISIBLE_DEVICES=1 python evaluation.py --model_name PPO_speed_"$i" --checkpoint_path /app/gearformer_model/models/ --encoder_checkpoint_name Xtransformer_0.0001_18_encoder.dict --decoder_checkpoint_name PPO_speed_"$i"_decoder.dict --val_data_path /app/simulator/test_data.csv --BS 1024
done
