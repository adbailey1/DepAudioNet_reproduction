import numpy as np
import pickle
import csv
import os
import h5py


def csv_read(file, start=None, end=None):
    """
    Read a csv (comma separated value) file and append each line to a list

    Input:
        file: The location of the csv file
        start: Start location for a read line
        end: End location for a read line

    Output:
        data: List of each row from csv file
    """
    data = []
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if start is not None and end is not None:
                    data.append(row[start:end])
                else:
                    data.append(row)
    return data


def load_data(path, labels):
    """
    Loads specific data from a dataset using indexes from labels.

    Input:
        path: The location to the database
        labels: The database labels which include the indexes of the specific
                data to load

    Output:
        features: The dataset features
    """
    with h5py.File(path, 'r') as h5:
        features = h5['features'][:]

    features = features[labels[-1].tolist()]

    return features

DATASET = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/datasets/DAIC-WOZ'
WORKSPACE_MAIN_DIR = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2'
WORKSPACE_FILES_DIR = os.path.join('/home', 'andrew', 'PycharmProjects',
                                   'depaudionet')
TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')

dset = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2/raw_exp/complete_database.h5'

data1 = csv_read(TRAIN_SPLIT_PATH)
data2 = csv_read(DEV_SPLIT_PATH)

train_labels = np.array([[i[0] for i in data1], [i[1] for i in data1],
                         [i[2] for i in data1],
                         [i[3] for i in data1]]).astype(int)

dev_labels = np.array([[i[0] for i in data2], [i[1] for i in data2],
                       [i[2] for i in data2],
                       [i[3] for i in data2]]).astype(int)

path = TEST_SPLIT_PATH
data3 = csv_read(path)
test_labels = np.array([[i[0] for i in data3], [i[1] for i in data3],
                        [i[2] for i in data3],
                        [i[3] for i in data3]]).astype(int)

data = [j for i in [data1, data2, data3] for j in i]
data.sort()

train = np.zeros((train_labels.shape[1])).astype(int)
dev = np.zeros((dev_labels.shape[1])).astype(int)
test = np.zeros((test_labels.shape[1])).astype(int)
for p, i in enumerate(data):
    h = np.where(train_labels[0] == int(i[0]))
    if len(h[0]) == 0:
        h = np.where(dev_labels[0] == int(i[0]))
        if len(h[0]) == 0:
            pass
        else:
            dev[h[0][0]] = p
    else:
        train[h[0][0]] = p

train_labels = np.concatenate((train_labels, train.reshape(1, -1)))
dev_labels = np.concatenate((dev_labels, dev.reshape(1, -1)))

train_features = load_data(dset, train_labels)
dev_features = load_data(dset, dev_labels)

train_length = dev_length = 0

for i in train_features:
    train_length += i[0].shape[0] / 16000
for i in dev_features:
    dev_length += i[0].shape[0] / 16000

train_length /= 3600
dev_length /= 3600

print(f"Training set contains {train_length} hours")
print(f"Validation set contains {dev_length} hours")

s = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2/raw_exp/summary' \
    '.pickle'

f = pickle.load(open(s, 'rb'))
min_length = f[1][f[0].index('MinSamples')]

depnet_length = (min_length / (16000 * 3600)) * (2 * 558)

print(f"DepAudioNet's Training Length is {depnet_length} hours ")

