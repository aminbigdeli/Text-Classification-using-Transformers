#Define the path of each parameter
CUDA_VISIBLE_DEVICES=1 \
python train.py \
     -model bert-base-uncased \
     -train ${train_path} \
     -dev ${dev_path} \
     -res ${result_path}
     -max_sequence_len 64\
     -epoch 10 \
     -train_batch_size 16\
     -valid_batch_size 16 \
     -lr 2e-5 \
     -n_warmup_steps 0 \