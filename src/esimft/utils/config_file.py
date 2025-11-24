import argparse


def config():
    parser = argparse.ArgumentParser()
    # gearformer settings
    parser.add_argument('--model_name', type=str, default="GearFormer")
    parser.add_argument('--catalogue_path', type=str, default="/app/dsl/catalogue.json", help='path to the catalogue.json')
    parser.add_argument('--language_path', type=str, default="/app/dsl/language.json", help='path to the language.json')
    parser.add_argument('--dim', type=int, default=512, help = "specify the dimention for xtransformer")
    parser.add_argument('--depth', type=int, default=6, help = "specify the depth for xtransformer")
    parser.add_argument('--head', type=int, default=8, help = "specify the number of head in xtransformer decoder")
    parser.add_argument('--WWL', type=float, default=1, help = "specify the weight for sequence's weight in the loss")
    parser.add_argument('--if_weight', type=bool, default=False, help = "pass weight as input to the model if true - it's always false in current version")
    parser.add_argument('--input_size', type=int, default=8)
    parser.add_argument('--output_size', type=int, default=53)
    parser.add_argument('--max_length', type=int, default=21)

    # gearformer data
    parser.add_argument('--gearformer_val_data', type=str, default="/app/data/gearformer_data/val_data.csv")
    parser.add_argument('--gearformer_test_data', type=str, default="/app/data/gearformer_data/test_data.csv")
    parser.add_argument('--gf_data_req_input_start_idx', type=int, default=2)
    parser.add_argument('--gf_data_req_input_end_idx', type=int, default=9)
    parser.add_argument('--gf_data_req_input_motion_type_idx', type=int, default=2)
    parser.add_argument('--gf_data_req_output_motion_type_idx', type=int, default=3)
    parser.add_argument('--gf_data_req_speed_idx', type=int, default=4)
    parser.add_argument('--gf_data_req_pos_idx', type=int, nargs="+", default=[5, 6, 7])
    parser.add_argument('--gf_data_req_output_motion_dir_idx', type=int, default=8)
    parser.add_argument('--gf_data_req_output_motion_sign_idx', type=int, default=9)
    parser.add_argument('--gf_data_req_price_idx', type=int, default=11)
    parser.add_argument('--gf_data_req_bb_volume_idx', type=int, default=12)

    # esimft data 
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--data_esimft_1', type=str, default="/app/data/esimft_data/esimft_1.pkl")
    parser.add_argument('--data_esimft_2', type=str, default="/app/data/esimft_data/esimft_2.pkl")
    parser.add_argument('--data_simft_test', type=str, default="/app/data/esimft_data/simft_test.pkl")
    parser.add_argument('--data_pareto_test', type=str, default="/app/data/esimft_data/pareto_test.pkl")
    parser.add_argument('--data_ric_aug', type=str, default="/app/data/esimft_data/ric_aug.pkl")
    parser.add_argument('--data_simft_test_aug', type=str, default="/app/data/esimft_data/simft_test_aug.pkl")
    parser.add_argument('--data_pareto_test_aug', type=str, default="/app/data/esimft_data/pareto_test_aug.pkl")
    parser.add_argument('--data_pareto_problems', type=str, default="/app/data/esimft_data/pareto_problems.pkl")
    parser.add_argument('--data_pareto_samples_folder', type=str, default="/app/data/esimft_data/pareto_samples")
    parser.add_argument('--data_sft_train', type=str)
    parser.add_argument('--data_sft_val', type=str)
    parser.add_argument('--data_pref_train', type=str)
    parser.add_argument('--data_pref_val', type=str)
    parser.add_argument('--sample_size', type=int, default=10)
    parser.add_argument('--simft_test_ratio', type=float, default=0.1)
    parser.add_argument('--sft_val_ratio', type=float, default=0.1)
    parser.add_argument('--pref_val_ratio', type=float, default=0.1)

    # checkpoints
    parser.add_argument('--checkpoint_path', type=str, default="/app/checkpoints", help = "path to the checkpoint folder")
    parser.add_argument('--gearformer_encoder_checkpoint_name', type=str, default="GearFormer_0.0001_19_encoder.dict")
    parser.add_argument('--gearformer_decoder_checkpoint_name', type=str, default="GearFormer_0.0001_19_decoder.dict")
    parser.add_argument('--sft_model_checkpoint_name', type=str)
    parser.add_argument('--dpo_model_checkpoint_folder', type=str)
    parser.add_argument('--ppo_model_checkpoint_folder', type=str)
    ## specify best models (for reward soups and pareto exp)
    parser.add_argument('--speed_model_checkpoint_name', type=str)
    parser.add_argument('--pos_model_checkpoint_name', type=str)
    parser.add_argument('--price_model_checkpoint_name', type=str)
    parser.add_argument('--bb_model_checkpoint_name', type=str)

    # mode params
    parser.add_argument('--req_name', type=str, default="speed", choices=["speed", "pos", "price", "bb"])
    parser.add_argument('--aug_data_type', type=str, default="ric", choices=["ric", "pareto_test", "simft_test"])
    parser.add_argument('--sft_mode', type=str, default="original_req", choices=["baseline", "original_req", "new_req", "ric"])

    # training params
    parser.add_argument('--epoch', type=int, default=100, help = "number of epochs")
    parser.add_argument('--BS', type=int, default=64, help = "specify the batch size")
    parser.add_argument('--lr', type=float, default=0.0001 , help = "learning rat for optimizer")

    parser.add_argument('--dpo_beta', type=float, default=0.1)

    parser.add_argument('--ppo_temperature', type=float, default=1.0)
    parser.add_argument('--ppo_beta', type=float, default=0.1)
    parser.add_argument('--ppo_clip', type=float, default=0.2)
    parser.add_argument('--ppo_mb_size', type=int, default=8)

    # etc
    parser.add_argument('--num_threads_sim', type=int, default=10)

    # reward soups weights
    parser.add_argument('--test_scenarios', type=str, nargs="+", default=["speed_pos", "speed_price", "speed_bb", "pos_price", "pos_bb", "price_bb", 
                 "speed_pos_bb", "speed_pos_price", "speed_bb_price", "pos_price_bb"])
    parser.add_argument('--two_reqs_weights_1', type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--two_reqs_weights_2', type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    parser.add_argument('--three_reqs_weights_1', type=float, nargs="+", default=[0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.33])
    parser.add_argument('--three_reqs_weights_2', type=float, nargs="+", default=[0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.33])
    parser.add_argument('--three_reqs_weights_3', type=float, nargs="+", default=[1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.33])

    # pareto exp
    parser.add_argument('--pareto_num_samples', type=int, default=30)
    parser.add_argument('--pareto_exp_num_problems', type=int, default=30)
    parser.add_argument('--pareto_exp_data_path', type=str, help = "path to the pkl file generated by pareto_exp.py")
    parser.add_argument('--ref_pareto_speed', type=float, default=18690.3)
    parser.add_argument('--ref_pareto_pos', type=float, default=1.34936)
    parser.add_argument('--ref_pareto_price', type=float, default=2374.11)
    parser.add_argument('--ref_pareto_bb', type=float, default=0.262308)
    parser.add_argument('--test_methods', type=str, nargs="+", default=["base", "sim", "eps", "eps_sim", "soup", "ric"])

    args = parser.parse_args()
    return args
