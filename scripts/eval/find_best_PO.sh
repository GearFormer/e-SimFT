for ((i=0; i<=${1}; i++)); do
    echo "price, DPO $i"
    python eval_simft.py --req_name "price" --sft_mode "new_req" --sft_model_checkpoint_name "dpo_price/epoch_${i}.dict"
done

for ((i=0; i<=${1}; i++)); do
    echo "bb, DPO $i"
    python eval_simft.py --req_name "bb" --sft_mode "new_req" --sft_model_checkpoint_name "dpo_bb/epoch_${i}.dict"
done

for ((i=0; i<=${1}; i++)); do
    echo "price, PPO $i"
    python eval_simft.py --req_name "price" --sft_mode "new_req" --sft_model_checkpoint_name "ppo_price/epoch_${i}.dict"
done

for ((i=0; i<=${1}; i++)); do
    echo "bb, PPO $i"
    python eval_simft.py --req_name "bb" --sft_mode "new_req" --sft_model_checkpoint_name "ppo_bb/epoch_${i}.dict"
done