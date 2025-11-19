# e-SimFT: Alignment of Generative Models with Simulation Feedback for Pareto-Front Design Exploration

**This work demonstrates fine-tuning of GearFormer with respect to prioritized design requirements and leveraging them to support Pareto-front design expoloration**

Link to paper: https://arxiv.org/abs/2502.02628


### 0. Pre-requisite:

```
git clone git@github.com:GearFormer/e-SimFT.git
```

Download checkpoints `*.dict` from https://github.com/GearFormer/GearFormer/tree/main/gearformer_model/checkpoints and place them in `./checkpoints`

Download `test_data.csv` and `val_data.csv` from https://github.com/GearFormer/GearFormer/tree/main/dataset and place them in `./data/gearformer_data`

Build and launch the docker container (change the Dockerfile base image as required):

```
cd e-SimFT
docker build -t esimft .
docker run --rm -it --name esimft --gpus all -v ./:/app esimft
```

### 1. Using the original validation and test portions of the original GearFormer dataset, prepare the dataset for e-SimFT.

```
python scripts/datagen/prepare_esimft_data.py
```

This will create 4 new dataset files (pkl) in `./data/esimft_data/`

### 2. Create SFT data for original requirements.

```
python scripts/datagen/gen_sft_data_original_req.py --req_name speed --sample_size 100 --data_sft_train data/esimft_data/sft_speed_train.pkl --data_sft_val data/esimft_data/sft_speed_val.pkl 

python scripts/datagen/gen_sft_data_original_req.py --req_name pos --sample_size 100 --data_sft_train data/esimft_data/sft_pos_train.pkl --data_sft_val data/esimft_data/sft_pos_val.pkl 
```

(needs use a large sample size to sample enough data that meet the original requirement targets)

### 3. Create SFT data for new requirements.

```
python scripts/datagen/gen_sft_data_new_req.py --req_name price --data_sft_train data/esimft_data/sft_price_train.pkl --data_sft_val data/esimft_data/sft_price_val.pkl 

python scripts/datagen/gen_sft_data_new_req.py --req_name bb --data_sft_train data/esimft_data/sft_bb_train.pkl --data_sft_val data/esimft_data/sft_bb_val.pkl 
```

### 4. Create preference data for new requirements.

```
python scripts/datagen/gen_pref_data.py --req_name price --data_pref_train data/esimft_data/pref_price_train.pkl --data_pref_val data/esimft_data/pref_price_val.pkl 

python scripts/datagen/gen_pref_data.py --req_name bb --data_pref_train data/esimft_data/pref_bb_train.pkl --data_pref_val data/esimft_data/pref_bb_val.pkl 
```

### 5. (for benchmarking) Create rewards-in-context training data.

```
python scripts/datagen/aug_data.py --aug_data_type ric

python scripts/datagen/gen_ric_data.py
```

### 6. (for benchmarking) Prepare Pareto problems and sampling strategies.

```
python scripts/datagen/aug_data.py --aug_data_type pareto_test

python scripts/datagen/prepare_pareto_problems.py --pareto_exp_num_problems 30

python scripts/datagen/prepare_pareto_samples.py --pareto_exp_num_problems 30
```

### 7. SFT the pre-trained model w.r.t. original/new requirements.

```
python scripts/train/train_sft.py --sft_mode original_req --req_name speed --data_sft_train data/esimft_data/sft_speed_train.pkl --data_sft_val data/esimft_data/sft_speed_val.pkl --lr 0.000001 --sft_model_checkpoint_name sft_speed.dict

python scripts/train/train_sft.py --sft_mode original_req --req_name pos --data_sft_train data/esimft_data/sft_pos_train.pkl --data_sft_val data/esimft_data/sft_pos_val.pkl --lr 0.000001 --sft_model_checkpoint_name sft_pos.dict

python scripts/train/train_sft.py --sft_mode new_req --req_name price --data_sft_train data/esimft_data/sft_price_train.pkl --data_sft_val data/esimft_data/sft_price_val.pkl --lr 0.00001 --sft_model_checkpoint_name sft_price.dict

python scripts/train/train_sft.py --sft_mode new_req --req_name bb --data_sft_train data/esimft_data/sft_bb_train.pkl --data_sft_val data/esimft_data/sft_bb_val.pkl --lr 0.00001 --sft_model_checkpoint_name sft_bb.dict
```

### 8. DPO the SFT model w.r.t. new requirements.

```
python scripts/train/train_dpo.py --data_pref_train data/esimft_data/pref_price_train.pkl --data_pref_val data/esimft_data/pref_price_val.pkl --epoch 20 --sft_model_checkpoint_name sft_price.dict --dpo_model_checkpoint_folder dpo_price

python scripts/train/train_dpo.py --data_pref_train data/esimft_data/pref_bb_train.pkl --data_pref_val data/esimft_data/pref_bb_val.pkl --epoch 20 --sft_model_checkpoint_name sft_bb.dict --dpo_model_checkpoint_folder dpo_bb
```

### 9. PPO the SFT model w.r.t. new requirements.

```
python scripts/train/train_ppo.py --data_pref_train data/esimft_data/pref_price_train.pkl --data_pref_val data/esimft_data/pref_price_val.pkl --epoch 20 --sft_model_checkpoint_name sft_price.dict --ppo_model_checkpoint_folder ppo_price

python scripts/train/train_ppo.py --data_pref_train data/esimft_data/pref_bb_train.pkl --data_pref_val data/esimft_data/pref_bb_val.pkl --epoch 20 --sft_model_checkpoint_name sft_bb.dict --ppo_model_checkpoint_folder ppo_bb
```

### 10. To evaluate the baseline and SFT models w.r.t. original requirements:

```
python eval_or.py --req_name "[pos/speed]"

python eval_or.py --decoder_checkpoint_name "SFT_[pos/speed]_decoder.dict" --req_name "[pos/speed]"

```

### 11. To evaluate the baseline, SFT models, and DPO/PPO models w.r.t. new requirements:

```
python aug_data.py --aug_data_type simft_test

python eval_baseline_nr.py --req_name "[price/bb]"

python eval_simft_nr.py --decoder_checkpoint_name "SFT_[price/bb]_decoder.dict" --req_name "[price/bb]"

python eval_simft_nr.py --decoder_checkpoint_name "[DPO/PPO]_[price/bb]_[i]_decoder.dict" --req_name "[price/bb]"
```

### 12. To find the best performing DPO/PPO models:

```
./find_best_PO.sh
```

### 13. (for benchmarking) Train rewards-in-context model

```
python scripts/train/train_sft.py --sft_mode ric --lr 0.000001
```

### 14. (for benchmarking) Create rewarded-soup models. 

```
python soup.py --decoder_price_checkpoint_name "[the best model found in step 12]" --decoder_bb_checkpoint_name "[the best model found in step 12]"
```

### 15. (for benchmarking) Generate Pareto fronts

```
python pareto_exp.py --N 30
python pareto_exp.py --N 300
```

### 16. (for benchmarking) Evaluate Pareto fronts

```
python pareto_eval.py --pareto_exp_data_path esimft_data/pareto_data_30.pkl
python pareto_eval.py --pareto_exp_data_path esimft_data/pareto_data_300.pkl
```
