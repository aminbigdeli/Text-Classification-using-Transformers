import argparse
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from collections import defaultdict
from torch import cuda
from tqdm import tqdm
from datetime import datetime
import os 
from data_loader import Dataset, DataLoader

device = 'cuda' if cuda.is_available() else 'cpu'

def prepare_result_dir(model, result_dir):
  try:
    if len(result_dir) == 0:
      if not os.path.exists('experiments'):
        os.makedirs('experiments')
      else:
        result_dir = "experiments"
      now = datetime.now()
      directory = now.strftime("%Y-%m-%d_%H%M%S_experiment_"+str(model))
      path = os.path.join(result_dir, directory)
      os.mkdir(path)
    else:
      path = result_dir
    return path
  except FileNotFoundError as e:
    print(e)

#Training function
def train(model, data_loader, optimizer, device, scheduler, n_examples):
  print("Training the Model")
  model = model.train()
  losses = []
  correct_predictions = 0
  for data in tqdm(data_loader):
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    targets = data["target"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      labels = targets
    )
    _, preds = torch.max(outputs[1], dim=1)  # the second return value is logits
    loss = outputs[0] #the first return value is loss
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

#Evaluation function - used when adopting K-fold
def eval(model, data_loader, device, n_examples):
  print("Validating the Model")
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for data in tqdm(data_loader):
      input_ids = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      targets = data["target"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels = targets
      )
      _, preds = torch.max(outputs[1], dim=1)
      loss = outputs[0]
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

#Prediction function - used to calculate the accuracy of the model when true labels are available
def get_predictions(model, data_loader):
  print("Testing the Best-Perfomred Model")
  model = model.eval()
  sequences = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for data in tqdm(data_loader):
      texts = data["sequence"]
      input_ids = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      targets = data["target"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
	      labels = targets
      )
      _, preds = torch.max(outputs[1], dim=1)
      sequences.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs[1])
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return sequences, predictions, prediction_probs, real_values

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-model', type=str, default='bert-base-uncased')
  parser.add_argument('-train', type=str, default='')
  parser.add_argument('-dev', type=str, default='')
  parser.add_argument('-max_sequence_len', type=int, default=64)
  parser.add_argument('-epoch', type=int, default=10)
  parser.add_argument('-train_batch_size', type=int, default=16)
  parser.add_argument('-valid_batch_size', type=int, default=16)
  parser.add_argument('-res', type=str, default='')
  parser.add_argument('-lr', type=float, default=2e-5)
  parser.add_argument('-n_warmup_steps', type=int, default=0)
  args = parser.parse_args()
  args.model = args.model.lower()

  res_path = prepare_result_dir(args.model, args.res)
  logger = open(res_path + "/log.txt", "w")
  logger.write("Model: " + args.model + "\n")
  
  tokenizer = AutoTokenizer.from_pretrained(args.model)
  
  train_set = Dataset(args.train, tokenizer, args.max_sequence_len)
  classes, encoded_classes, train_set_shape = train_set.get_info()
  logger.write("Label Encoding: " + str(classes) + "-->" + str(np.sort(encoded_classes)) + "\n")
  encoded_classes = encoded_classes.astype(str)
  logger.write("shape of the train set: {} \n".format(train_set_shape))
  train_data_loader = DataLoader(train_set, args.train_batch_size, shuffle  = False, num_workers = 0)

  model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels = len(encoded_classes)) 
  model = model.to(device)

  if len(args.dev) != 0:
    dev_set = Dataset(args.dev, tokenizer, args.max_sequence_len)
    _, _, dev_set_shape = dev_set.get_info()
    df_dev = dev_set.get_dataframe()
    logger.write("shape of the dev set: {} \n".format(dev_set_shape))
    valid_data_loader = DataLoader(dev_set, args.valid_batch_size, shuffle  = False, num_workers = 0)

  optimizer = AdamW(params =  model.parameters(), lr = args.lr)
  total_steps = len(train_data_loader) * args.epoch
  scheduler = get_linear_schedule_with_warmup(
              optimizer,
              num_warmup_steps = args.n_warmup_steps,
              num_training_steps = total_steps
          )

  history = defaultdict(list)
  best_accuracy = best_epoch = 0
  for epoch in range(args.epoch):
    logger.write(f'Epoch {epoch + 1}/{args.epoch}')
    print(f'Epoch {epoch + 1}/{args.epoch}')
    logger.write("\n")
    train_acc, train_loss = train(
                model,
                train_data_loader,
                optimizer,
                device,
                scheduler,
                train_set.__len__()
        )
    logger.write(f'Train loss {train_loss} accuracy {train_acc}')
    logger.write("\n")
    if len(args.dev) != 0:
      val_acc, val_loss = eval(
                model,
                valid_data_loader,
                device,
                dev_set.__len__()
          )
      logger.write(f'Val   loss {val_loss} accuracy {val_acc}')
      logger.write("\n")
      logger.write('-' * 10)
      logger.write("\n")
      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)
      if val_acc > best_accuracy:
        torch.save(model.state_dict(), res_path + "/best_performed_fine_tuend_" + args.model + ".bin")
        best_accuracy = val_acc
        best_epoch = epoch + 1

  print("Training Process Finished!")
  
  if len(args.dev) == 0:
    torch.save(model.state_dict(), res_path + "/fine_tuend_" + args.model + ".bin")
    print("Model saved successfully!")
  else:
    logger.write("\n")
    logger.write("Best Epoch: {}".format(best_epoch))
    logger.write("\n")
    logger.write("Best Accuracy: {}".format(best_accuracy))
    logger.write("\n")

    model.load_state_dict(torch.load(res_path + "/best_performed_fine_tuend_" + args.model + ".bin"))
    model = model.to(device)
    _, y_pred, y_pred_probs, y_test = get_predictions(
            model,
            valid_data_loader
            )
    df_dev['prediction'] = y_pred
    for i in range(len(encoded_classes)):
      df_dev[encoded_classes[i]] = y_pred_probs[:, i]
    df_dev.to_csv(res_path + "/predictions.tsv", sep = "\t", index = False)
    logger.write('-' * 10)
    logger.write("\n")
    accuracy = accuracy_score(y_test, y_pred)
    print("Best Accuracy on the Validation set: {} \n".format(accuracy))
    logger.write("Classification Report: \n")
    logger.write(classification_report(y_test, y_pred, target_names = encoded_classes))
    logger.write("\n")
    logger.write('-' * 10)
    logger.write("\n")
    logger.write("Accuracy on the Validation set: {} \n".format(accuracy))
    logger.write("\n")
  logger.close()

  
if __name__ == "__main__":
  main()
