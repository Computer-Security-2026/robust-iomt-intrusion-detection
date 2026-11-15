# test_data_loader.py - Automatic checks for src/data_loader.py, using small fake CSV files.
# Run it:  python -m pytest tests/ -v

import os
import numpy as np
import pandas as pd
from src.data_loader import get_attack_name, load_data
def make_fake_data(folder):
    # Create fake train or test CSV files that look like the real ones
    column_names = ["feature_" + str(i) for i in range(45)]
    for split in ["train", "test"]:
        os.makedirs(os.path.join(folder, split))
        for name in ["Benign", "ARP_Spoofing", "Recon-Port_Scan"]:
            table = pd.DataFrame(np.random.rand(100, 45), columns=column_names)
            table.iloc[0, 0] = np.inf    # put in one infinity
            table.iloc[1, 1] = np.nan    # put in one missing
            file_name = name + "_" + split + ".pcap.csv"
            table.to_csv(os.path.join(folder, split, file_name), index=False)
# tmp_path is a temporary folder that pytest creates for us and deletes afterward.
def test_45_features(tmp_path):
    make_fake_data(tmp_path)
    data = load_data(tmp_path)
    assert data["X_train"].shape[1] == 45
    assert data["X_test"].shape[1] == 45
def test_labels_are_only_0_and_1(tmp_path):
    make_fake_data(tmp_path)
    data = load_data(tmp_path)
    assert set(data["y_train"]) == {0, 1}
    assert set(data["y_test"]) == {0, 1}
def test_no_bad_values_left(tmp_path):
    make_fake_data(tmp_path)
    data = load_data(tmp_path)
    assert not np.isnan(data["X_train"]).any()
    assert not np.isnan(data["X_test"]).any()
    assert not np.isinf(data["X_train"]).any()
    assert not np.isinf(data["X_test"]).any()
def test_holdout_is_removed_from_training_only(tmp_path):
    make_fake_data(tmp_path)
    data = load_data(tmp_path, holdout="ARP_Spoofing")
    assert "ARP_Spoofing" not in data["attack_train"]
    assert "ARP_Spoofing" in data["attack_test"]
def test_benign_rows_are_only_benign(tmp_path):
    make_fake_data(tmp_path)
    data = load_data(tmp_path)
    number_of_benign = (data["y_train"] == 0).sum()
    assert len(data["X_train_benign"]) == number_of_benign
def test_attack_name_comes_from_file_name():
    assert get_attack_name("data/train/Benign_train.pcap.csv") == "Benign"
    assert get_attack_name("data/test/Recon-Port_Scan_test.pcap.csv") == "Recon-Port_Scan"
