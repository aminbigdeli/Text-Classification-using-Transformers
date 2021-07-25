# Text Classification using Transformers
This repository contains the code and resources for text classification task using state-of-the-art Transformer models such as BERT and XLNet. You can use this github repository to fine-tune different tranformer models on your dataset for the task of sequence classification.
## Experiment
For the experiment part, we propose a Query Gender classifier that is capable of identifying the gender of each sentence. As the first step and in order to be able to label sequences based on their gender at scale, we employed the [gender-annotated dataset](https://github.com/aminbigdeli/Text-Classification-using-Transformers/blob/master/data/gender_annotated_dataset.tsv) released by [Navid Rekabsaz](https://github.com/navid-rekabsaz/GenderBias_IR) to train relevant classifiers. This dataset consists of 742 female, 1,202 male and 1,765 neutral queries. We trained various types of  classifiers on this dataset and in order to evaluate the performance of the classifiers, we adopt a 5-fold cross-validation strategy.
<table class="tg">
<thead>
  <tr>
    <th class="tg-fymr" rowspan="2">Classifier</th>
    <th class="tg-fymr" rowspan="2">Accuracy</th>
    <th class="tg-fymr" colspan="3">F1-Score</th>
  </tr>
  <tr>
    <td class="tg-fymr"><b>Female</td>
    <td class="tg-fymr"><b>Male</td>
    <td class="tg-fymr"><b>Neutral</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-xnov">BERT (base uncased)</td>
    <td class="tg-oyjm"><b>0.856</td>
    <td class="tg-oyjm"><b>0.816</td>
    <td class="tg-oyjm"><b>0.872</td>
    <td class="tg-oyjm"><b>0.862</td>
  </tr>
  <tr>
    <td class="tg-xnov">DistilBERT (base uncased)</td>
    <td class="tg-xnov">0.847</td>
    <td class="tg-xnov">0.815</td>
    <td class="tg-xnov">0.861</td>
    <td class="tg-xnov">0.853</td>
  </tr>
  <tr>
    <td class="tg-xnov">RoBERTa</td>
    <td class="tg-xnov">0.810</td>
    <td class="tg-xnov">0.733</td>
    <td class="tg-xnov">0.820</td>
    <td class="tg-xnov">0.836</td>
  </tr>
  <tr>
    <td class="tg-xnov">DistilBERT (base cased)</td>
    <td class="tg-xnov">0.800</td>
    <td class="tg-xnov">0.730</td>
    <td class="tg-xnov">0.823</td>
    <td class="tg-xnov">0.833</td>
  </tr>
  <tr>
    <td class="tg-xnov">BERT (base cased)</td>
    <td class="tg-xnov">0.797</td>
    <td class="tg-xnov">0.710</td>
    <td class="tg-xnov">0.805</td>
    <td class="tg-xnov">0.827</td>
  </tr>
  <tr>
    <td class="tg-xnov">XLNet (base cased)</td>
    <td class="tg-xnov">0.795</td>
    <td class="tg-xnov">0.710</td>
    <td class="tg-xnov">0.805</td>
    <td class="tg-xnov">0.826</td>
  </tr>
</tbody>
</table>

You can find the fine-tuned BERT model on the gender-annotated dataset [here](https://drive.google.com/file/d/1k-XL9xpdmzlXmVMs0MRrQv6PChQW8Vfn/view?usp=sharing) and use inference.py for predicting the gender of sequences.

## Models
All of the models on the [Huggingface](https://huggingface.co/transformers) that support `AutoModelForSequenceClassification` are supported by this repository and can be used by setting the model parameter of the train.py with the appropriate name of the model. Some of them are listed below and the others can be found on Huggingface website.
```
Models = {
    "BERT base uncased": "bert-base-uncased",
    "BERT base cased": "bert-base-cased",
    "BERT large uncased": "bert-large-uncased",
    "BERT large cased": "bert-large-cased",
    "XLNet": "xlnet-base-cased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased"
}

```
## Train
In order to fine-tune each of the transformer models on your dataset, you can execute the following bash file:
```shell
bash train.sh
```
Please note that before executing the bash file, you need to define a set of files path in it.

#### Option
```
-model                     BERT, XLNet, RoBERTa, DistilBERT, etc.
-train                     path to training dataset.
-dev                       path to validation dataset if you want to validate the trained model, otherwise ignore it.
-res                       path to result dir. If not defined, the model will be saved in the experiments dir.
-max_sequence_len          max length of sequence tokens.
-epoch                     number of epochs.
-train_batch_size          batch size.
-valid_batch_size          batch size.
-lr                        learning rate.
-n_warmup_steps            warmup steps.
```
## Inference
In order to inference the fine-tuned models, you can execute the following bash file:
```shell
bash inference.sh
```
Please note that before executing the bash file, you need to define a set of files path in it.

### Option
```
-model                     BERT, XLNet, RoBERTa, DistilBERT, etc.
-test                      path to test dataset.
-checkpoint                path to checkpoint.
-res                       path to result dir.
-num_labels                number of classes.
-max_sequence_len          max length of sequence tokens.
-test_batch_size           batch size.
```

## Data Format
Both train.py and inference.py scripts receive the datasets in .tsv format with the following format:
|file|format (.tsv)|
|:---|:-----|
|train| id \t sequence \t label|
|dev  | id \t sequence \t label|
|test | id \t sequence|

## Cross Validation
In order to compare the performance of different transformer models on your dataset, you can perform k-fold cross validation by executing the following bash file:
```shell
bash cross_validation.sh
```
By default, dataset is the [gender-annotated dataset](https://github.com/aminbigdeli/Text-Classification-using-Transformers/blob/master/data/gender_annotated_dataset.tsv) and the result path is [/data/cross_validation_path](https://github.com/aminbigdeli/Text-Classification-using-Transformers/tree/master/data/cross_validation_path) folder.
