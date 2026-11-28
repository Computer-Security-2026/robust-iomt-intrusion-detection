#data_loader.py - Part 1: load, label, clean, and scale the CICIoMT2024 data.
#Author: Joseph Andrews (CS 4371 Group 1)
#Labels come from the file name: Benign = 0, any attack = 1.
#Run it: python -m src.data_loader
#Use it in code:  from src.data_loader import load_data, data = load_data(), data = load_data(holdout="ARP_Spoofing")

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#The same random seed is used everywhere so results come out the same every run
SEED = 42
#Some files have hundreds of thousands of rows. We only keep up to this many
#rows per file so the code runs fast on a laptop.
MAX_ROWS_PER_FILE = 50000
def get_attack_name(file_path):
    #Get the attack name from the file name
    file_name = os.path.basename(file_path)
    file_name = file_name.replace("_train.pcap.csv", "")
    file_name = file_name.replace("_test.pcap.csv", "")
    return file_name                                 # 'ARP_Spoofing'
def read_folder(folder):
    #Read every CSV in a folder into one table
    #Adds two columns attack and label
    all_tables = []
    for file_path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        table = pd.read_csv(file_path)
        #If the file is very big, keep a random sample of rows.
        if len(table) > MAX_ROWS_PER_FILE:
            table = table.sample(n=MAX_ROWS_PER_FILE, random_state=SEED)
        attack = get_attack_name(file_path)
        table["attack"] = attack
        if attack == "Benign":
            table["label"] = 0
        else:
            table["label"] = 1
        all_tables.append(table)
    if len(all_tables) == 0:
        raise FileNotFoundError("No CSV files found in " + folder)
    #Stack all the tables on top of each other into one big table
    return pd.concat(all_tables, ignore_index=True)
def load_data(data_folder="data", holdout=None):
    #Load scale the train and test data for the models
    #An attack name to remove from training only
    #Returns a dictionary with X_train, y_train, X_test, y_test, X_train_benign, attack_test, and feature_names
    if not os.path.isdir(os.path.join(data_folder, "train")):
        print("Full data not found, using data/sample instead.")
        data_folder = os.path.join(data_folder, "sample")
    #step 1 read the CSV files
    train = read_folder(os.path.join(data_folder, "train"))
    test = read_folder(os.path.join(data_folder, "test"))
    #step 2 if we are holding out an attack, remove it from training
    if holdout is not None:
        if holdout not in train["attack"].unique():
            raise ValueError("Holdout name not found: " + holdout)
        train = train[train["attack"] != holdout]
    #step 3 split each table into features (X) and answers (y).
    #X = the 45 traffic numbers y = the 0/1 label
    #The label must never be inside X or the model would be cheating
    feature_names = [col for col in train.columns if col not in ("attack", "label")]
    X_train = train[feature_names].copy()
    X_test = test[feature_names].copy()
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    #step 4 clean bad values.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    #The medians come from the training data and information from the test set would leak into training
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)
    #step 5 scale the numbers.
    #The columns are on very different scales
    #StandardScaler puts every column on a similar scale so no column dominates
    #Again the scaler LEARNS from training data only, then is applied to both.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # learn from train, then scale train
    X_test = scaler.transform(X_test)         # scale test using what was learned from train
    #step 6 package everything up for teammates.
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_benign": X_train[y_train == 0],     # benign rows only
        "attack_train": train["attack"].to_numpy(),
        "attack_test": test["attack"].to_numpy(),
        "feature_names": feature_names,
        "scaler": scaler,
        "train_medians": train_medians,
        "holdout": holdout,
    }
    print_summary(data)
    return data
def print_summary(data):
    #print a short report
    print("------------------------------------------")
    print("Held-out attack:", data["holdout"])
    print("Number of features:", len(data["feature_names"]))
    print("Training rows:", len(data["X_train"]))
    print("Test rows:", len(data["X_test"]))
    print("Benign training rows (for Isolation Forest):", len(data["X_train_benign"]))
    print()
    print("Training rows per class:")
    print(pd.Series(data["attack_train"]).value_counts().to_string())
    print()
    print("Test rows per class:")
    print(pd.Series(data["attack_test"]).value_counts().to_string())
    print()
    print("Bad values left in X_train:", np.isnan(data["X_train"]).sum())
    print("Bad values left in X_test:", np.isnan(data["X_test"]).sum())
    print("------------------------------------------")
#This part only runs when you type:  python -m src.data_loader
if __name__ == "__main__":
    print("NORMAL RUN (all attacks in training)")
    load_data()
    print()
    print("HOLDOUT RUN (ARP_Spoofing hidden from training)")
    load_data(holdout="ARP_Spoofing")
