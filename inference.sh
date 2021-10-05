#Define the path of each parameter
CUDA_VISIBLE_DEVICES=1 \
python inference.py \
     -model bert-base-uncased \
     -test ${test_path} \
     -checkpoint ${checkpoint_path} \
     -res ${result_path} \
     -num_labels 3 \
     -max_sequence_len 64 \
     -test_batch_size 16 
