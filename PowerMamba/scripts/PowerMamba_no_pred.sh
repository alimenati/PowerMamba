if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./csv_results" ]; then
    mkdir ./csv_results
fi
if [ ! -d "./results" ]; then
    mkdir ./results
fi
if [ ! -d "./test_results" ]; then
    mkdir ./test_results
fi

model_name=PowerMamba
root_path_name=../data/
data_path_name=price_no_pred.csv
features=Mm
c_out=22
include_pred=0
target_name=0 # only important in features = 'S' mode!
model_id_name=price
data_name=custom


for pred_len in 24
do

start_time=$(date "+%Y%m%d_%H%M%S")
log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${fc_drop}_${batch_size}_${fc_dropout}_${kernel_size}_${start_time}.log"

python -u run_longExp.py \
--random_seed 2024 \
--is_training 1 \
--root_path $root_path_name \
--data_path $data_path_name \
--fc_dropout 0.2 \
--head_dropout 0 \
--target $target_name \
--model_id ${model_id_name}_${seq_len}_${pred_len} \
--model $model_name \
--data $data_name \
--features $features \
--seq_len 240 \
--pred_len $pred_len \
--enc_in $c_out \
--dec_in $c_out \
--c_out $c_out \
--n_embed 300 \
--dropout 0.2 \
--revin 1 \
--ch_ind 0 \
--residual 1 \
--individual 0 \
--dconv 2 \
--d_state 256 \
--e_fact 1 \
--des 'Exp' \
--kernel_size 7 \
--train_epochs 1 \
--lradj '5' \
--include_pred $include_pred \
--itr 1 \
--batch_size 512 \
--learning_rate 0.001 >> $log_file

done
