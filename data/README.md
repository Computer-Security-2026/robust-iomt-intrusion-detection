# Dataset
We are using the CICIoMT2024 dataset. This is the same dataset the paper used.
It has network traffic from medical IoT devices, some normal and some attacks.
The full data files are too big to put on GitHub, so we only uploaded a small
sample in the data/sample folder.
## Where to get the data
Go to http://cicresearch.ca/IOTDataset/CICIoMT2024/ and open the folder
WiFi_and_MQTT/attacks/CSV. We downloaded the files on November 14, 2026.
We used 6 files:
- Benign, normal traffic
- ARP Spoofing
- Recon Port Scan
Put the train files in data/train and the test files in data/test.
## What our code does to the data
Our file src/data_loader.py gets the data ready for the models:
1. It gives each row a label from the file name. Normal traffic is 0 and attacks are 1.
2. It only keeps up to 50,000 rows from each file so it runs faster.
3. It fixes any bad or missing values.
4. It puts all the numbers on the same scale.
For steps 3 and 4 we only use the training data, so the test data stays unseen
and the test is fair.
## Testing an unknown attack
We can hide ARP Spoofing from training by using load_data.
That way the models only see ARP Spoofing for the first time during testing,
which lets us check if they can catch an attack they were never trained on.
## Checksums
These codes let someone check that they downloaded the exact same files we used.
f0d68653cfe5873f280a0cd84323145d6112861819204b22e3d1e690d02df6ad  data/train/ARP_Spoofing_train.pcap.csv
5337565793ff0460270da0b848e2ca8eafbd529a9715ce33f8ba4578a772fc29  data/train/Benign_train.pcap.csv
7dc0785d3e69d4e1a083582dad0efc4c94c35bbc75e56ec0ddb9bd2f6a34e872  data/train/Recon-Port_Scan_train.pcap.csv
a897da3919604aefb392ce88493441d3a871ac3174d95345f6c4d43dfa847a4a  data/test/ARP_Spoofing_test.pcap.csv
bfbc0299d0c0c7a2e13c17aa513b15cceeef57034173104ba2309c3557154f83  data/test/Benign_test.pcap.csv
d7b7d49d68fb6b1efe14ca5a5343d23c6e3ab00e86b0c64880ef99935f291b06  data/test/Recon-Port_Scan_test.pcap.csv