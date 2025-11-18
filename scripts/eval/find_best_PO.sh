#!/bin/bash

for i in {0..19}
do
    echo "price, DPO $i" 
    CUDA_VISIBLE_DEVICES=0 python eval_simft_nr.py --decoder_checkpoint_name "DPO_price_"$i"_decoder.dict" --req_name "price"
done

for i in {0..19}
do
    echo "bb, DPO $i" 
    CUDA_VISIBLE_DEVICES=0 python eval_simft_nr.py --decoder_checkpoint_name "DPO_bb_"$i"_decoder.dict" --req_name "bb"
done

for i in {0..19}
do
    echo "price, PPO $i" 
    CUDA_VISIBLE_DEVICES=0 python eval_simft_nr.py --decoder_checkpoint_name "PPO_price_"$i"_decoder.dict" --req_name "price"
done

for i in {0..19}
do
    echo "bb, PPO $i" 
    CUDA_VISIBLE_DEVICES=0 python eval_simft_nr.py --decoder_checkpoint_name "PPO_bb_"$i"_decoder.dict" --req_name "bb"
done

