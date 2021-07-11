# Text Classification using Transformers
This repository contains the code and resources for text classification task using state-of-the-art Transformer models such as BERT and XLNet. You can use this github repository to fine-tune different tranformer models on your dataset for the task of sequence classification.
## Experiment
For the experiment part, we propose a Query Gender classifier that is capable of identifying the gender of each sentence. As the first step and in order to be able to label sequences based on their gender at scale, we employed the [gender-annotated dataset](https://github.com/aminbigdeli/Text-Classification-using-Transformers/blob/master/data/gender_annotated_dataset.tsv) released by [Navid Rekabsaz](https://github.com/navid-rekabsaz/GenderBias_IR) to train relevant classifiers. This dataset consists of 742 female, 1,202 male and 1,765 neutral queries. We trained various types of  classifiers on this dataset and in order to evaluate the performance of the classifiers, we adopt a 5-fold cross-validation strategy.
<table>
<thead>
  <tr>
    <th style="text-align: right;" class="tg-0lax">Transformer Model</th>
    <th class="tg-0lax">Classifier</th>
    <th class="tg-0lax">Accuracy</th>
    <th class="tg-baqh" colspan="3">F1-Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">BERT (base uncased)</td>
    <td class="tg-l2oz"><b>0.856</td>
    <td class="tg-l2oz"><b>0.816</td>
    <td class="tg-l2oz"><b>0.872</td>
    <td class="tg-l2oz"><b>0.862</td>
  </tr>
  <tr>
    <td class="tg-0lax">DistilBERT (base uncased)</td>
    <td class="tg-lqy6">0.847</td>
    <td class="tg-lqy6">0.815</td>
    <td class="tg-lqy6">0.861</td>
    <td class="tg-lqy6">0.853</td>
  </tr>
  <tr>
    <td class="tg-0lax">RoBERTa</td>
    <td class="tg-lqy6">0.810</td>
    <td class="tg-lqy6">0.733</td>
    <td class="tg-lqy6">0.820</td>
    <td class="tg-lqy6">0.836</td>
  </tr>
  <tr>
    <td class="tg-0lax">DistilBERT (base cased)</td>
    <td class="tg-lqy6">0.800</td>
    <td class="tg-lqy6">0.730</td>
    <td class="tg-lqy6">0.823</td>
    <td class="tg-lqy6">0.833</td>
  </tr>
  <tr>
    <td class="tg-0lax">BERT (base cased)</td>
    <td class="tg-lqy6">0.797</td>
    <td class="tg-lqy6">0.710</td>
    <td class="tg-lqy6">0.805</td>
    <td class="tg-lqy6">0.827</td>
  </tr>
  <tr>
    <td class="tg-0lax">XLNet (base cased)</td>
    <td class="tg-lqy6">0.795</td>
    <td class="tg-lqy6">0.710</td>
    <td class="tg-lqy6">0.805</td>
    <td class="tg-lqy6">0.826</td>
  </tr>
</tbody>
</table>

## Train
In order to fine-tune each of the transformer models on your dataset, you can execute the following bash file:
```shell
bash train.sh
```
Please note that before executing the bash file, you need to define a set of path files in it.

#### Option
```
-model                     BERT, XLNet, RoBERTa, DistilBERT, etc.
-train                     Path to training dataset.
-dev                       Path to validation dataset if you want to validate the trained model, otherwise ignore it.
-res                       Path to result dir. If not defined, the model will be saved in the experiments dir.
-max_sequence_len          Max length of sequence tokens.
-epoch                     Number of epochs.
-train_batch_size          Batch size.
-valid_batch_size          Batch size.
-lr                        Learning rate.
-n_warmup_steps            Warmup steps.
```
## Inference
In order to inference the fine-tuned models, you can execute the following bash file:
```shell
bash inference.sh
```
Please note that before executing the bash file, you need to define a set of path files in it.

### Option
```
-model                     BERT, XLNet, RoBERTa, DistilBERT, etc.
-test                      Path to test dataset.
-checkpoint                Path to checkpoint.
-res                       Path to result dir.
-num_labels                Number of classes.
-max_sequence_len          Max length of sequence tokens.
-test_batch_size           Batch size.
```

## Data Format
Both train.py and inference.py scripts receive the datasets in .tsv format with the following columns:
|file|format (.tsv)|
|:---|:-----|
|train| "id" \t "sequence" \t "label"|
|dev  | "id" \t "sequence" \t "label"|
|test | "id" \t "sequence"|

## Cross Validation
In order to compare the performance of different transformer models on your dataset, you can perform k-fold cross validation by executing the following bash file:
```shell
bash cross_validation.sh
```
By default, dataset is the [gender-annotated dataset](https://github.com/aminbigdeli/Text-Classification-using-Transformers/blob/master/data/gender_annotated_dataset.tsv) and the result path is [/data/cross_validation_path](https://github.com/aminbigdeli/Text-Classification-using-Transformers/tree/master/data/cross_validation_path) folder.
