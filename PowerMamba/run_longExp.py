import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import logging
from fractions import Fraction


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_energy_data(energy_data_str):
    energy_data_dict = {}
    items = energy_data_str.split(',')
    for item in items:
        key, value1, value2 = item.split(':')
        energy_data_dict[key] = [int(value1), int(value2)]
    return energy_data_dict

def parse_col_info_data(col_info_data_str):
    data_dict = {}
    items = col_info_data_str.split(',')
    for item in items:
        key, value1, value2 = item.split(':')
        value1 = int(value1)
        value2 = int(value2)
        data_dict[key] = [value1, value2]
    return data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # RANDOM SEED
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # BASIC CONFIG
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [TimeMachine]')
    parser.add_argument('--model_id_name', type=str, required=False, default='custom', help='model id name')

    # DATALOADER
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--project_dict', type=str, required=False, default=' ', help='dataset type')
    parser.add_argument('--col_info_dict', type=str, required=False, default=' ', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS, Mm]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    parser.add_argument('--n_pred_col', type=int, default='1',
                        help='#column for prediction')
    
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')




    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')

    parser.add_argument('--dec_in', type=int, default=23, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=23, help='output size')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    
    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.2, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # FORECASTING TASK
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--n1',type=int,default=256,help='First Embedded representation')
    parser.add_argument('--n2',type=int,default=128,help='Second Embedded representation')
    parser.add_argument('--n_embed',type=int,default=256,help='Embedding dimension for GridTimes')


    # METHOD
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--ch_ind', type=int, default=1, help='Channel Independence; True 1 False 0')
    parser.add_argument('--residual', type=int, default=1, help='Residual Connection; True 1 False 0')
    parser.add_argument('--d_state', type=int, default=256, help='d_state parameter of Mamba')
    parser.add_argument('--dconv', type=int, default=2, help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact', type=int, default=1, help='expand factor parameter of Mamba')
    parser.add_argument('--enc_in', type=int, default=23, help='encoder input size') #Use this hyperparameter as the number of channels
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    # OPTIMIZATION
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    parser.add_argument('--include_pred', type=int, default=0, help='any prediction in dataset?')
    # parser.add_argument('--n_nonpred_col', type=int, default=26, help='n_nonpred_col')

    args = parser.parse_args()
    args.target = args.target.split(',')
    args.project_dict = parse_energy_data(args.project_dict)
    args.col_info_dict = parse_col_info_data(args.col_info_dict)


    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.model_id_name=args.data_path[:-4]


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_n1{}_n2{}_dr{}_cin{}_rin{}_res{}_dst{}_dconv{}_efact{}_run_{}'.format(
                args.model_id,
                args.model,
                args.model_id_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.n1,
                args.n2,
                args.n_embed,
                args.dropout,
                args.ch_ind,
                args.revin,
                args.residual,
                args.d_state,
                args.dconv,
                args.e_fact,
            )
            print(setting)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_n1{}_n2{}_dr{}_cin{}_rin{}_res{}_dst{}_dconv{}_efact{}'.format(
                args.model_id,
                args.model,
                args.model_id_name,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.n1,
                args.n2,
                args.n_embed,
                args.dropout,
                args.ch_ind,
                args.revin,
                args.residual,
                args.d_state,
                args.dconv,
                args.e_fact)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        
