from sklearn.model_selection import KFold
import pandas as pd
import argparse
import os 


def split_dataset(ds_path, cross_validation_path, k_fold):
    df = pd.read_csv(ds_path, sep = "\t")
    df = df.values
    
    kf = KFold(n_splits = k_fold, random_state = 0, shuffle = True) 
    fold_idx = 0
    for train_index, dev_index in kf.split(df):
        fold_idx +=1
        X_train, X_dev = df[train_index], df[dev_index]
        X_train = pd.DataFrame(X_train)
        X_dev = pd.DataFrame(X_dev)
        if not os.path.exists(cross_validation_path + "/fold-" + str(fold_idx)):
          os.makedirs(cross_validation_path + "/fold-" + str(fold_idx))
        train_path = cross_validation_path + "/fold-" + str(fold_idx) + "/train.tsv"
        dev_path = cross_validation_path + "/fold-" + str(fold_idx) + "/test.tsv"
        X_train.to_csv(train_path, sep = "\t", header = None, index=False)
        X_dev.to_csv(dev_path, sep = "\t", header = None, index=False)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset_path', type=str, default='')
  parser.add_argument('-cross_validation_path', type=str, default='')
  parser.add_argument('-k_fold', type=int, default=5)
  args = parser.parse_args()

  if len(args.cross_validation_path) !=0 :
    split_dataset(args.dataset_path, args.cross_validation_path, args.k_fold)   
  else:
      print("You should have specified the cross validation path!")

if __name__ == "__main__":
    main()

