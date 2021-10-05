#bash
# You need to specify these configurations
dataset_path=./data/gender_annotated_dataset.tsv
cross_validation_folder=./data/cross_validation_path
k_fold=5
# End here

#Spliting the dataset into defined K-folds 
#Comment this section if you have your dataset splitted into K-Folds
python split.py \
 -dataset_path ${dataset_path} \
 -cross_validation_path ${cross_validation_folder} \
 -k_fold ${k_fold}
# End here

#K-Fold cross validation
for ((idx=1; idx<=$k_fold; idx++)); do
    echo "--------------------------- Fold ${idx} ---------------------------"

    train_path=${cross_validation_folder}/fold-${idx}/train.tsv
    dev_path=${cross_validation_folder}/fold-${idx}/test.tsv
    result_path=${cross_validation_folder}/fold-${idx}/

    python train.py \
     -model bert-base-uncased \
     -train ${train_path} \
     -dev ${dev_path} \
     -res ${result_path} \
     -max_sequence_len 64\
     -epoch 10 \
     -train_batch_size 16 \
     -valid_batch_size 16 \
     -lr 2e-5 \
     -n_warmup_steps 0 
     
done

#Done
