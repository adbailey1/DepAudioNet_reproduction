import numpy as np
from exp_run import dataset_processing
import time
import utilities


def create_folds_test(dp):
    number_zeros = [8, 9, 10, 12, 127]
    number_ones = [8, 10, 10, 34, 49]
    data_folds = [4, 4, 5, 5, 4]
    expected_zeros = [[2, 2, 2, 2], [3, 2, 2, 2], [2, 2, 2, 2, 2], [3, 3, 2, 2, 2], [32, 32, 32, 31]]
    expected_ones = [[2, 2, 2, 2], [3, 3, 2, 2], [2, 2, 2, 2, 2], [7, 7, 7, 7, 6], [13, 12, 12, 12]]

    for i in range(len(number_ones)):
        folds_zeros = dp.create_data_folds(data_folds[i], number_zeros[i])
        folds_ones = dp.create_data_folds(data_folds[i], number_ones[i])
        print('Expected Folds of Zeros:', expected_zeros[i], 'Received Values:', folds_zeros)
        print('Expected Folds of Ones:', expected_ones[i], 'Received Values:', folds_ones)

    if folds_zeros != expected_zeros[i] and folds_ones != expected_ones[i]:
        return False
    return True


def count_classes_test():
    test_zeros = 8
    test_ones = 10
    classes = [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]
    indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    test_index_zero = []
    test_index_one = []

    for pointer, i in enumerate(classes):
        if i == 0:
            test_index_zero.append(indexes[pointer])
        else:
            test_index_one.append(indexes[pointer])

    zeros, index_zeros, ones, index_ones = utilities.count_classes(classes)

    if zeros != test_zeros or ones != test_ones:
        return False
    if index_zeros != test_index_zero or index_ones != test_index_one:
        return False
    else:
        return True


def find_files_test(dp):
    complete_database = np.array([[300, 301, 302, 303, 304],
                                  [0, 0, 1, 1, 0],
                                  [0, 1, 2, 3, 4]])
    train_set = [[303, 300, 304], [1, 0, 0]]
    expected = np.array([[303, 300, 304], [1, 0, 0], [3, 0, 4]])

    new_set = dp.find_files_in_database(complete_database, train_set)

    if np.array_equal(expected, new_set):
        return True
    else:
        return False


def run_tests_dp(config, current_path, config_path):
    dp = dataset_processing.DataProcessor('_', '_', '_', '_', current_path,
                                          'sub', '_', True, False, config_path)
    start = time.time()
    passed = find_files_test(dp)
    end = time.time()
    if passed:
        print(f"Test mod_audio_test Passed in {end - start}s")
    else:
        print(f"Test mod_audio_test Failed in {end - start}s")

    start = time.time()
    passed = count_classes_test()
    end = time.time()
    if passed:
        print(f"Test count_classes_test Passed in {end - start}s")
    else:
        print(f"Test count_classes_test Failed in {end - start}s")

    start = time.time()
    passed = create_folds_test(dp)
    end = time.time()
    if passed:
        print(f"Test count_classes_test Passed in {end - start}s")
    else:
        print(f"Test count_classes_test Failed in {end - start}s")
