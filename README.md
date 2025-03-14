# e-SimFT: Alignment of Generative Models with Simulation Feedback for Pareto-Front Design Exploration

**This work demonstrates fine-tuning of GearFormer with respect to prioritized design requirements and leveraging them to support Pareto-front design expoloration**

Link to paper: https://arxiv.org/abs/2502.02628

1. Using the original validation and test portions of the original GearFormer dataset, prepare the dataset for e-SimFT.

```
python prepare_simft_data.py
```

This will create 4 new dataset files (pkl) in `/esimft_data`

2. Create SFT data for original requirements.

```
python gen_sft_data_[pos/speed].py
```

3. Create SFT data for new requirements.

```
python gen_sft_data_nr.py
```

4. Create preference data for new requirements.

```
python gen_pref_data.py
```

5. (for benchmarking) Create rewards-in-context training data.

```
python aug_pref_data.py
python gen_ric_data.py
```

6. (for benchmarking) Prepare Pareto problems and sampling strategies.

```
python aug_pareto_data.py
python prepare_pareto_problems.py
python prepare_pareto_samples.py --N 30
```

7. SFT the pre-trained model w.r.t. original requirements.

```
CUDA_VISIBLE_DEVICES=0 python -m train_models.train_sft --train_data_path "esimft_data/sft_[pos/speed]_train.pkl" --val_data_path "esimft_data/sft_[pos/speed]_val.pkl" --BS 64 --lr 0.000001 --req_name "[pos/speed]"
```

8. SFT the pre-trained model w.r.t. new requirements.

```
CUDA_VISIBLE_DEVICES=0 python -m train_models.train_sft_nr --train_data_path "esimft_data/sft_obj_train.pkl" --val_data_path "esimft_data/sft_obj_val.pkl" --BS 64 --lr 0.00001 --req_name "[price/bb]"
```


9. DPO the SFT model w.r.t. new requirements.

```
CUDA_VISIBLE_DEVICES=0 python -m train_models.train_dpo --train_data_path "esimft_data/pref_[price/bb]_train.pkl" --val_data_path "esimft_data/pref_[price/bb]_val.pkl" --epoch 20 --BS 64 --lr 0.000001 --req_name "[price/bb]"
```

10. PPO the SFT model w.r.t. new requirements.

```
CUDA_VISIBLE_DEVICES=0 python -m train_models.train_ppo --train_data_path "esimft_data/pref_[price/bb]_train.pkl" --val_data_path "esimft_data/pref_[price/bb]_val.pkl" --epoch 20 --BS 64 --lr 0.00001 --decoder_checkpoint_name "SFT_[price/bb]_decoder.dict" --req_name "[price/bb]"
```

11. To evaluate the baseline, SFT models, and DPO/PPO models w.r.t. new requirements:

```
python aug_test_data.py

CUDA_VISIBLE_DEVICES=0 python eval_baseline_nr.py --req_name "[price/bb]"

CUDA_VISIBLE_DEVICES=0 python eval_simft_nr.py --decoder_checkpoint_name "SFT_[price/bb]_decoder.dict" --req_name "[price/bb]"

CUDA_VISIBLE_DEVICES=0 python eval_simft_nr.py --decoder_checkpoint_name "[DPO/PPO]_[price/bb]_[i]_decoder.dict" --req_name "[price/bb]"
```

12. To find the DPO/PPO models:

```
./find_best_PO.sh
```

price: DPO-12, req met 0.7485714285714286, validity 0.9510869565217391

price: PPO-9, req met 0.726775956284153, validity 0.9945652173913043

bb: DPO-11, req met 0.7921348314606742, validity 0.967391304347826

bb: PPO-5, req met 0.6179775280898876, validity 0.967391304347826

13. (for benchmarking) Train rewards-in-context model

```
CUDA_VISIBLE_DEVICES=0 python -m train_models.train_ric --train_data_path "esimft_data/ric_train.pkl" --val_data_path "esimft_data/ric_val.pkl" --epoch 20 --BS 64 --lr 0.000001
```

14. (for benchmarking) Create rewarded-soup models. 

```
python soup.py --decoder_price_checkpoint_name "[the best model found in step 12]" --decoder_bb_checkpoint_name "[the best model found in step 12]"
```

15. (for benchmarking) Generate Pareto fronts

```
python pareto_exp.py
```

16. (for benchmarking) Evaluate Pareto fronts

```
python pareto_eval.py
```


Copyright 2025, Autodesk, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software on a non-commercial research or educational basis only, the rights to use, copy, modify, merge, publish, distribute, and/or sublicense the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

![image](https://github.com/user-attachments/assets/f46a3919-2f3d-49f0-8199-d2c2bdcb115f)
