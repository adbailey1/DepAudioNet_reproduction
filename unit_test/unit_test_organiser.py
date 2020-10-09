import logging
import os
import pickle
import utilities
from data_loader import organiser
from unit_test import config_test
import numpy as np


def determine_fold_test():
    gt_dev = [1, 2]
    gt_train = [[2, 3, 4], [1, 3, 4, 5]]
    total_folds = [4, 5]
    current_fold = [1, 2]
    output_dev = []
    output_train = []
    for i in range(len(gt_dev)):
        fold_train, fold_dev = organiser.determine_folds(current_fold[i],
                                                         total_folds[i])
        if fold_dev != gt_dev[i]:
            print(f"Fail: Expected Folds for Dev: {gt_dev[i]} but Recevied:"
                  f" {fold_dev}")
            return False
        for j in range(len(fold_train)):
            if fold_train[j] != gt_train[i][j]:
                print(f"Fail Expected Folds for Train: {gt_train[i]} but "
                      f"Received: {fold_train}")
                return False

        output_dev.append(fold_dev)
        output_train.append(fold_train)

    print(f"Test Passed Trainig Folds: Expected: {gt_train} and Received:"
          f" {output_train}")
    print(f"Test Passed Dev Folds: Expected: {gt_dev} and Received:"
          f" {output_dev}")
    return True


def find_weight_test():
    zeros = [100, 200, 300]
    ones = [100, 100, 400]
    gt = [[1, 1], [0.5, 1], [1, .75]]
    output = []
    for i in range(len(gt)):
        weights = organiser.find_weight(zeros[i], ones[i])
        if weights != gt[i]:
            print(f"Fail: Expected weights: {gt[i]} but Recevied: {weights}")
            return False
        output.append(weights)

    print(f"Test Passed: Expected: {gt} and Received: {output}")
    return True


def updated_features_test(main_logger):
    # Folder - Class - Score
    test1 = [300, 1, 11]
    test2 = [301, 0, 4]
    test3 = [302, 1, 16]
    test4 = [303, 0, 1]
    meta = [test1, test2, test3, test4]
    label = 'train'
    min_samples = [0, 0, 400, 400]
    mel_bins = [100, 100, 100, 100]
    amcs = [True, True, False, False]
    segment_dim = [100, 99, 100, 99]
    convert_to_image = False
    data = np.random.randn(480000)
    ground_truth = [[48, 100, 100], [49, 100, 99], [4, 100, 100], [5, 100, 99]]
    for i in range(len(meta)):
        feature, folder, classes, score, ind = organiser.get_updated_features(data,
                                           min_samples[i], label, meta[i],
                                           main_logger, segment_dim[i],
                                           mel_bins[i], amcs[i],
                                           convert_to_image)
        expected_dim = ground_truth[i]
        expected_meta = meta[i]
        x1, y1, z1 = feature.shape
        if expected_dim[0] != x1 or expected_dim[1] != y1 or expected_dim[2] != z1:
            print('Fail: Feature dimensions not expected.')
            print(f"Dimensions Received: {x1}, {y1}, {z1}, Expected:"
                  f" {expected_dim}")
            return False
        if len(folder) != expected_dim[0] or len(classes) != expected_dim[0] or \
                len(score) != expected_dim[0]:
            print('Fail: Length of folders/classes/scores not expected')
            print(f"Lengths Received: Folder: {len(folder)}, Class:"
                  f"{len(classes)}, Score: {len(score)}")
            print(f"Lengths Expected: Folder/Class/Score: {expected_dim[0]}")
            return False
        for j in range(len(folder)):
            if folder[j] != expected_meta[0] or classes[j] != expected_meta[
                1] or score[j] != expected_meta[2]:
                print(f"Fail: Values Received not expected")
                print(f"Received: Folder:{folder[j]}, Class: {classes[j]}, "
                      f"Score:{score[j]}")
                print(f"Expected: FOlder:{expected_meta[0]}, Class:{expected_meta[1]}, Score"
                      f"{expected_meta[2]}")
                return False


    print('Test Passed')
    return True


def calculate_length_test():
    # with h5py.File(database, 'r') as h5:
    #     ground_truth = h5['features'][:]
    data = np.random.randn(400000, 1)
    min_samples = 400
    label = 'train'
    segment_dim = [100, 101, 99, 100, 99]
    mel_bins = 100
    acms = [True, True, True, False, False]
    ground_truth = [40, 40, 41, 4, 5]
    output = []
    for i in range(len(segment_dim)):
        temp_length = organiser.calculate_length(data, min_samples, label,
                                                 segment_dim[i], mel_bins,
                                                 acms[i])
        output.append(temp_length)
        if temp_length != ground_truth[i]:
            print('Fail: Expected,', ground_truth, 'Received', output)
            return False

    print('Pass: Expected,', ground_truth, 'Received', output)
    return True


def seconds_test():
    window_size = [1024, 1024]
    hop = [512, 0]
    overlap = [0, 30]
    sample_rate = 16000
    seconds = 30
    ground_truth = [938, 670]
    samples1 = utilities.seconds_to_sample(seconds, window_size[0],
                                           hop_length=hop[0],
                                           sample_rate=sample_rate)
    samples2 = utilities.seconds_to_sample(seconds, window_size[1],
                                           overlap=overlap[1],
                                           sample_rate=sample_rate)
    samples = [samples1, samples2]
    if ground_truth == samples:
        print('Pass: Expected,', ground_truth, 'Received', samples)
        return True
    else:
        print('Fail: Expected,', ground_truth, 'Received', samples)
        return False


def run_tests_o(config):
    current_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(current_path, 'log.log')
    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    main_handler = logging.handlers.RotatingFileHandler(log_path)
    main_logger.addHandler(main_handler)
    database = os.path.join(config_test.DATASET, 'complete_database.h5')
    summary_file = os.path.join(
        config_test.WORKSPACE_MAIN_DIR, config_test.FOLDER_NAME,
        'summary.pickle')
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    result = seconds_test()
    result = calculate_length_test(main_logger)
    result = updated_features_test()
    result = find_weight_test()
    result = determine_fold_test()
    os.remove(log_path)

