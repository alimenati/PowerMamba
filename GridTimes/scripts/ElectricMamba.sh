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


declare -A project_dict_=(['COAST']='241 97' ['EAST']='242 49' ['FWEST']='243 73' ['NORTH']='244 1'
                         ['NCENT']='245 25' ['SOUTH']='246 169' ['SCENT']='247 121' ['WEST']='248 145'
                         ['SOLAR']='254 193' ['WIND']='253 217')

project_dict=""
keys=(${!project_dict_[@]})
for i in "${!keys[@]}"; do
    key=${keys[$i]}
    values=(${project_dict_[$key]})
    first_value=${values[0]}
    second_value=${values[1]}
    project_dict+="$key:$first_value:$second_value"
    if [ "$i" -lt "$((${#keys[@]} - 1))" ]; then
        project_dict+=","
    fi
done

declare -A col_info_dict_=( ['price']='-8 8' ['wind']='-10 1' ['solar']='-9 1' ['anc_serv']='-14 4' ['load']='-22 8' )
keys=(${!col_info_dict_[@]})

# Loop through the keys with index
for i in "${!keys[@]}"; do
    key=${keys[$i]}
    values=(${col_info_dict_[$key]})
    first_value=${values[0]}
    second_value=${values[1]}

    # Append to the string
    col_info_dict+="$key:$first_value:$second_value"

    # Add a comma if it's not the last element
    if [ "$i" -lt "$((${#keys[@]} - 1))" ]; then
        col_info_dict+=","
    fi
done



model_name=ElectricMamba
features=Mm
c_out=22
include_pred=0
n_nonpred_col=22

target_name='COAST','EAST','FWEST','NORTH','NCENT','SOUTH','SCENT','WEST','REGDN','REGUP','RRS','NSPIN','WIND_ACTUAL_SYSTEM_WIDE','SOLAR_ACTUAL_SYSTEM_WIDE','LZ_AEN','LZ_CPS','LZ_HOUSTON','LZ_LCRA','LZ_NORTH','LZ_RAYBN','LZ_SOUTH','LZ_WEST' 

root_path_name=../data/price
data_path_name=price.csv

model_id_name=price
data_name=custom

random_seed=2024

residual=1
fc_drop=0.2
dstate=256
dconv=2
e_fact=1
batch_size=512 
pred_len=24
seq_len=240
individual=0
fc_dropout=.2
n_embed=300
kernel_size=7
lradjval='5'
learning_rate=0.001


for pred_len in 24 48 72 96 168
do

start_time=$(date "+%Y%m%d_%H%M%S")
log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${fc_drop}_${batch_size}_${fc_dropout}_${kernel_size}_${rin}_${start_time}.log"

echo "Start Time: $(date "+%Y-%m-%d %H:%M:%S")" > $log_file

python -u run_longExp.py \
--random_seed $random_seed \
--is_training 1 \
--root_path $root_path_name \
--data_path $data_path_name \
--fc_dropout $fc_dropout \
--head_dropout 0 \
--target $target_name \
--model_id ${model_id_name}_${seq_len}_${pred_len} \
--model $model_name \
--data $data_name \
--features $features \
--seq_len $seq_len \
--pred_len $pred_len \
--enc_in $n_nonpred_col \
--dec_in $n_nonpred_col \
--c_out $c_out \
--project_dict $project_dict \
--col_info_dict $col_info_dict \
--n_embed $n_embed \
--dropout $fc_drop \
--revin 1 \
--ch_ind 0 \
--residual $residual \
--individual $individual \
--dconv $dconv \
--d_state $dstate \
--e_fact $e_fact \
--des 'Exp' \
--kernel_size $kernel_size \
--train_epochs 1 \
--lradj $lradjval \
--include_pred $include_pred \
--n_nonpred_col $n_nonpred_col \
--n_pred_col $c_out \
--itr 1 --batch_size $batch_size --learning_rate $learning_rate >> $log_file

# Log end time in the file
echo "End Time: $(date "+%Y-%m-%d %H:%M:%S")" >> $log_file
done
