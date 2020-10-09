import time
import data_gen
import numpy as np
import h5py
import os
import pickle
import logging
import importlib


def process_data_test(dg, database, labels, label, expected_dimensions):

    with h5py.File(database, 'r') as h5:
        ground_truth = h5['features'][:]
    feats, labels = dg.process_data(ground_truth, labels, label)
    if expected_dimensions == list(feats.shape):
        print('Pass: Expected,', expected_dimensions, 'Received',
              list(feats.shape))
        return True
    else:
        print('Fail: Expected,', expected_dimensions, 'Received',
              list(feats.shape))
        return False


def get_value_data(mel_bins, samples):
    random_file = np.random.randn(mel_bins, samples)
    random_file = random_file.astype(np.float32)
    random_file = np.reshape(random_file, mel_bins*samples)

    return random_file


def run_tests_dg(config, current_path):
    conf = importlib.import_module('config')
    mel_bins = conf.EXPERIMENT_DETAILS['MEL_BINS']
    labels = np.array([[300, 301, 303, 306], [1, 0, 0, 0], [0, 1, 2, 3]])
    samples = [20, 33, 24, 31]
    segments = 10
    summary_file = [['MaxSamples', 'MaxWindows', 'MinSamples', 'MinWindows',
                     'SampleRate', 'NumberFiles', 'ListOfSamples'],
                    [33792, 33, 20480, 20, 16000, 4, [20, 33, 24, 31]]]
    feature_overlap = [0, 0, 50]
    expected_dimensions = [[8, 128, 10], [13, 128, 10], [12, 128, 10]]
    concat_not_shorten = [False, True, False]
    train_file_path = os.path.join(current_path, 'train_meta.pickle')
    with open(train_file_path, 'wb') as f:
        pickle.dump(labels, f)
    dev_file_path = os.path.join(current_path, 'dev_meta.pickle')
    with open(dev_file_path, 'wb') as f:
        pickle.dump(labels, f)

    datatype = h5py.special_dtype(vlen=np.float32)
    database = os.path.join(current_path, 'complete_database.h5')
    h5file = h5py.File(database, 'w')
    num_files = len(labels[0])
    h5file.create_dataset(name='folder',
                          data=[i for i in labels[0]],
                          dtype=np.int16)
    h5file.create_dataset(name='class',
                          data=[i for i in labels[1]],
                          dtype=np.int8)
    h5file.create_dataset(name='index',
                          data=[i for i in labels[2]],
                          dtype=np.int16)
    h5file.create_dataset(name='features',
                          shape=(num_files, 1),
                          maxshape=(num_files, None),
                          dtype=datatype)

    for i in range(len(labels[0])):
        value = get_value_data(mel_bins, samples[i])
        h5file['features'][i] = value

    h5file.close()

    summary_file_path = os.path.join(current_path, 'summary.pickle')
    with open(summary_file_path, 'wb') as f:
        pickle.dump(summary_file, f)

    log_path = os.path.join(current_path, 'log.log')
    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.handlers.RotatingFileHandler(log_path)
    main_logger.addHandler(main_handler)

    for i in range(len(expected_dimensions)):
        dg = data_gen.GenerateData(32, train_file_path, dev_file_path,
                                   summary_file_path, database, segments,
                                   main_logger,
                                   config, amcns=concat_not_shorten[i],
                                   fop=feature_overlap[i])
        start = time.time()
        passed = process_data_test(dg, database, labels, 'train',
                                   expected_dimensions[i])
        end = time.time()
        if passed:
            print(f"Test check_folder_data_order Passed in {end - start}s")
        else:
            print(f"Test check_folder_data_order Failed in {end - start}s")
    os.remove(train_file_path)
    os.remove(dev_file_path)
    os.remove(database)
    os.remove(log_path)
    os.remove(summary_file_path)
