import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_path', type=str, default="/app/train_models/utils/language.json", help='path to the language.json')
    parser.add_argument('--catalogue_path', type=str, default="/app/train_models/utils/catalogue.json", help='path to the catalogue.json')
    parser.add_argument('--checkpoint_path', type=str, default="/app/train_models/models", help = "path to the checkpoint folder")
    parser.add_argument('--train_data_path', type=str, default="/app/simulator/val_data.csv", help = "path to the train data folder")
    parser.add_argument('--val_data_path', type=str, default="/app/simulator/val_data.csv", help = "path to the val or test data folder")
    parser.add_argument('--epoch', type=int, default=100, help = "number of epochs")
    parser.add_argument('--BS', type=int, default=64, help = "specify the batch size")
    parser.add_argument('--dim', type=int, default=512, help = "specify the dimention for xtransformer")
    parser.add_argument('--depth', type=int, default=6, help = "specify the depth for xtransformer")
    parser.add_argument('--head', type=int, default=8, help = "specify the number of head in xtransformer decoder")
    parser.add_argument('--WWL', type=float, default=1, help = "specify the weight for sequence's weight in the loss")
    parser.add_argument('--model_name', type=str, default="Xtransformer", help = "It should be Xtransformer")
    parser.add_argument('--lr', type=float, default=0.0001 , help = "learning rat for optimizer")
    parser.add_argument('--if_weight', type=bool, default=False, help = "pass weight as input to the model if true - it's always false in current version")
    parser.add_argument('--encoder_checkpoint_name', type=str, default="Xtransformer_0.0001_18_encoder.dict" , help = "name of the encoder checkpoint")
    parser.add_argument('--encoder_price_checkpoint_name', type=str, default="SFT_price_new_encoder.dict" , help = "name of the decoder checkpoint")
    parser.add_argument('--encoder_bb_checkpoint_name', type=str, default="SFT_bb_new_encoder.dict" , help = "name of the decoder checkpoint")
    parser.add_argument('--decoder_speed_checkpoint_name', type=str, default="SFT_speed_decoder.dict" , help = "name of the decoder checkpoint")
    parser.add_argument('--decoder_pos_checkpoint_name', type=str, default="SFT_pos_decoder.dict" , help = "name of the decoder checkpoint")
    parser.add_argument('--decoder_price_checkpoint_name', type=str, default="SFT_price_decoder.dict" , help = "name of the decoder checkpoint")
    parser.add_argument('--decoder_bb_checkpoint_name', type=str, default="SFT_bb_decoder.dict" , help = "name of the decoder checkpoint")

    parser.add_argument('--req_name', type=str, default="pos")
    args = parser.parse_args()
    return args
