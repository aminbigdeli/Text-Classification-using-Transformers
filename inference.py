import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm
from torch import cuda
from data_loader import Dataset, DataLoader

device = 'cuda' if cuda.is_available() else 'cpu'

def get_predictions(model, data_loader):
  model = model.eval()
  sequences = []
  predictions = []
  prediction_probs = []
  with torch.no_grad():
    for data in tqdm(data_loader):
      texts = data["sequence"]
      input_ids = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs[0], dim=1)
      sequences.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs[0])
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  return sequences, predictions, prediction_probs

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model', type=str, default='bert-base-uncased')
  parser.add_argument('-checkpoint', type=str, default='')
  parser.add_argument('-num_labels', type=int, default='', help="Number of classes of the fine-tuned model")
  parser.add_argument('-test', type=str, default='')
  parser.add_argument('-max_sequence_len', type=int, default=64)
  parser.add_argument('-test_batch_size', type=int, default=16)
  parser.add_argument('-res', type=str, default='')
  args = parser.parse_args()
  args.model = args.model.lower()

  if len(args.test) == 0:
    print("You should specify the path of the test set")
    exit()
  
  logger = open(args.res + "/log.txt", "w")
  logger.write("Model: " + args.model + "\n")
  logger.write("Path of the checkpoint used for predicting the test set: " + args.checkpoint + "\n")

  tokenizer = AutoTokenizer.from_pretrained(args.model)

  test_set = Dataset(args.test, tokenizer, args.max_sequence_len, True)
  _, _, test_set_shape = test_set.get_info()
  df_test = test_set.get_dataframe()
  logger.write("shape of the test set: {} \n".format(test_set_shape))
  test_data_loader = DataLoader(test_set, args.test_batch_size, shuffle = False, num_workers = 0)

  print("Loading the Model ...")
  model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = args.num_labels)
  model = model.to(device)
  model.load_state_dict(torch.load(args.checkpoint, map_location = device))
  print("The Model Loaded Successfully!")


  print("Classifying ...")
  _, y_pred, y_pred_probs = get_predictions(model, test_data_loader)
  pred_df = pd.DataFrame(df_test.values.tolist(), columns = ["id","sequence"])
  for i in range(args.num_labels):
    pred_df["weight_class_"+str(i)] = y_pred_probs[:, i]
  pred_df['predicted_label'] = y_pred
  pred_df.to_csv(args.res + "/classification_result.tsv", sep = "\t", index = False)
  print("Classification Finished!")

if __name__ == "__main__":
    main()

