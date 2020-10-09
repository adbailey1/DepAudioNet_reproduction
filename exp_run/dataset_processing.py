import os
import sys
import utilities.utilities_main as util
import pickle
import logging
import logging.handlers
import random
import math
import numpy as np
import h5py
from exp_run import config_dataset
from exp_run import config_1 as config


def create_data_arrays(ground_truth_data, shuffled_index):
    """
    Creates an array of meta data using shuffled indexes

    Inputs
        ground_truth_data: numpy.array - Holds the meta data for the dataset
        shuffled_index: list - Indexes of the sub dataset to be created

    Output
        meta: numpy.array - Meta data for the sub dataset
    """
    ground_truth_numpy = np.array(ground_truth_data)
    # Folder, Class, Score, Gender, Index
    meta = np.array([ground_truth_numpy[0][shuffled_index],
                     ground_truth_numpy[1][shuffled_index],
                     ground_truth_numpy[2][shuffled_index],
                     ground_truth_numpy[3][shuffled_index],
                     ground_truth_numpy[4][shuffled_index]])
    return meta


def logging_dataset_info(new_fold_index,  num_fold_one, num_fold_zero,
                         data_logger):
    """
    Logs the important information relating to the creation of the data folds
    such as the number of files in each fold

    Inputs
        new_fold_index: list - Combined index files for both classes
        num_fold_one: list - Holds the number of files for the class per fold
        num_fold_zero: list - Holds the number of files for the class per fold
        data_logger: logger - Records the information
        config: obj - The configuration file
    """
    data_logger.info(config_dataset.SEPARATOR)
    data_logger.info(f"The number of files in each of the folds are: ")
    num_files = []
    for i, data in enumerate(new_fold_index):
        num_files.append(len(data[0])+len(data[1]))
        data_logger.info(config_dataset.SEPARATOR)
        data_logger.info(f"Fold_{i}: {num_files[i]}, number of ones: "
                         f"{num_fold_one[i]}, number of zeros: "
                         f"{num_fold_zero[i]}")
        data_logger.info(config_dataset.SEPARATOR)


def create_data_folds(data_folds, start_num):
    """
    Works out how many files should be in each fold for a specific class

    Inputs
        data_folds: int - The number of folds for the current experiment
        start_num: int - The number of files for the current class

    Output
        preliminary: list - Contains the number of instances in each fold for
                     the specific class
    """
    preliminary = [math.floor(start_num / data_folds)] * data_folds
    leftover = start_num % data_folds

    for i in range(leftover):
        preliminary[i] += 1

    return preliminary


def create_meta_data(complete_data, fold_indexes, data_split_paths):
    """

    Inputs
        complete_data: numpy.array - Meta data for the complete dataset
        fold_indexes: list - Holds the indexes for each fold
        data_split_paths: list - Used to iterate through the different folds

    Outputs
        data_split: list - holds the numpy.arrays for the meta data of the folds
    """
    data_split = []
    for num in range(len(data_split_paths)):
        meta_data_numpy = create_data_arrays(complete_data, fold_indexes[num])
        data_split.append(meta_data_numpy)

    return data_split


def save_fold_information(data_split_paths, data_splits, workspace,
                          folder_name):
    """
    Inputs
        data_split_paths: list - List of strings for the folders for each fold
        data_splits: list - Holds numpy.array of the meta data for the
                     different splits
        workspace: str - Location of the experiment sub-directory
        folder_name: str - The name of the current fold folder
    """
    for i, name in enumerate(data_split_paths):
        new_save_location = os.path.join(workspace,
                                         folder_name,
                                         name + '.pickle')

        with open(new_save_location, 'wb') as f:
            pickle.dump(data_splits[i], f)


def create_data_sub_files(ground_truth_data, data_logger, folds, workspace,
                          test_set_for_comp=None, mode='sub', gender='-'):
    """
    Creates the folds for the current experiment. This works for both modes
    because in complete the test set is separate and in the sub mode,
    the test set is also separate. The difference is in how both modes split
    the data. In "complete" the only thing to do is split the data into
    train/dev whereas in "sub" we need to split the data into train/dev/test

    Inputs
        ground_truth_data: numpy.array - Holds meta data for the training
                           data including folder, class, score, and index
        data_logger: logger - Records important information
        folds: int - The total number of folds to split the experiment
        workspace: str - Location of the experiment sub-directory
        config: obj - The configuration file
        test_set_for_comp: Used for complete mode (not completed)
        mode: str - Set to complete if using the whole dataset with no
              validation or sub if using folds
    """
    comp_classes = ground_truth_data[1]
    zeros, index_zeros, ones, index_ones = util.count_classes(comp_classes)

    # Determine how many files to put into the train/dev/test sets
    if mode == 'complete':
        total_files_dataset = len(ground_truth_data[0]) + len(
            test_set_for_comp[0])
    else:
        num_fold_one = create_data_folds(folds, ones)
        num_fold_zero = create_data_folds(folds, zeros)

        folds_indexes = []
        folds_separated_indexes = []
        equal_zeros = []
        equal_fold = []
        for i in range(folds):
            if i is not folds - 1:
                new_fold_one = random.sample(index_ones, num_fold_one[i])
                new_fold_zero = random.sample(index_zeros, num_fold_zero[i])
                index_ones = [j for j in index_ones if
                              j not in new_fold_one]
                index_zeros = [j for j in index_zeros if
                               j not in new_fold_zero]
            else:
                new_fold_one = index_ones
                new_fold_zero = index_zeros

            equal_zeros.append(new_fold_zero[0:num_fold_one[i]])
            new_fold_separated_index = [new_fold_one, new_fold_zero]
            new_fold_index = [j for i in new_fold_separated_index for j in
                              i]
            new_equal_fold_separated = [new_fold_one, equal_zeros[i]]
            new_equal_fold = [j for i in new_equal_fold_separated for j in
                              i]
            folds_indexes.append(new_fold_index)
            folds_separated_indexes.append(new_fold_separated_index)
            equal_fold.append(new_equal_fold)

    logging_dataset_info(folds_separated_indexes, num_fold_one,
                         num_fold_zero, data_logger)

    data_split_paths = []
    for fold in range(1, folds+1):
        data_split_paths.append('Fold_'+str(fold))
    new_data_splits = create_meta_data(ground_truth_data,
                                       folds_indexes,
                                       data_split_paths)
    new_data_splits_equal = create_meta_data(ground_truth_data,
                                             equal_fold,
                                             data_split_paths)

    folder_name = 'data_folds_'+str(folds)
    if gender == 'm' or gender == 'f':
        folder_name = folder_name + '_' + gender
    save_fold_information(data_split_paths, new_data_splits, workspace,
                          folder_name)
    if test_set_for_comp is not None:
        save_fold_information(['test'], test_set_for_comp, workspace,
                              folder_name)
    folder_name = folder_name+'_equal'
    save_fold_information(data_split_paths, new_data_splits_equal, workspace,
                          folder_name)
    if test_set_for_comp is not None:
        save_fold_information(['test'], test_set_for_comp, workspace,
                              folder_name)


def find_files_in_database(complete_database, sub_section, test=False):
    """
    Find the location of the data split files (training or test) in the
    complete database

    Inputs
        complete_database: numpy.array - The complete meta data of the
                           database, folder, class, score, and index
        sub_section: list - folders, class, scores for the current set

    Output
        new_set: numpy.array - Full set with meta data including
                 folder, class, score, and index relating to original
                 location in the original database
    """
    indx = []
    for i in sub_section[0]:
        get_indexes = np.where(i == complete_database[0])
        indx.append(get_indexes[0].tolist())
    index = [j for i in indx for j in i]

    if test:
        # In test mode we only have 3 dimensions, folder ,gender and index,
        # the class and score are not provided
        new_set = np.zeros([complete_database.shape[0], len(index)]).astype(int)
        new_set[1:3, :] = new_set[1:3, :] - 1
        new_set[0, :] = complete_database[0][index]
        new_set[3, :] = complete_database[-2][index]
        new_set[4, :] = complete_database[-1][index]
    else:
        new_set = np.zeros([complete_database.shape[0], len(index)]).astype(int)
        new_set[:, :] = complete_database[0][index], \
                        complete_database[1][index], \
                        complete_database[2][index], \
                        complete_database[-2][index], \
                        complete_database[-1][index]

    return new_set


def partition_dataset(workspace, features_exp, features_directory, sub_dir,
                      current_directory, mode, dataset_path, folds=4,
                      gender='-'):
    """
    The top function that logs information regarding the experiment
    parameters, loads the database of features along with the labels in order
    to divide them into folds.

    Inputs
        workspace: str - Location of the experiment sub-directory
        features_exp: str - The type of features used in the experiment
        features_directory: str - Location of the features
        sub_dir: str - The folder of this current experiment
        current_directory: str - The current save location of this experiment
        mode: str - Set to complete if using the whole dataset or sub if
              using part in order to create a validation
        dataset_path: str - Location of the dataset files
        config: obj - The configuration file to be used for this experiment
        folds: int - The number of folds to split the dataset into
    """
    if gender == 'm' or gender == 'f':
        log_path = os.path.join(workspace, 'data_folds_'+str(folds)+'_'+gender,
                                'data_logger.log')
    else:
        log_path = os.path.join(workspace,
                                'data_folds_' + str(folds),
                                'data_logger.log')
    data_logger = logging.getLogger('DataLogger')
    data_logger.setLevel(logging.INFO)
    data_handler = logging.handlers.RotatingFileHandler(log_path)
    data_logger.addHandler(data_handler)
    data_logger.info(f"The workspace: {workspace}")
    data_logger.info(f"The experiment type is: {features_exp}")
    data_logger.info(f"The feature dir: {features_directory}")
    data_logger.info(f"The current experiment is: {sub_dir}")
    data_logger.info(f"The current directory is: {current_directory}")
    data_logger.info(f"The dataset mode is set to: {mode}")
    data_logger.info(f"The dataset dir: {dataset_path}")

    database_loc = os.path.join(features_directory, 'complete_database')
    if gender == 'm' or gender == 'f':
        database_loc = database_loc + '_' + gender + '.h5'
    else:
        database_loc = database_loc + '.h5'
    h = h5py.File(database_loc, 'r')
    complete_database_features = np.array([h['folder'][:],
                                           h['class'][:],
                                           h['score'][:],
                                           h['gender'][:],
                                           h['index'][:]]).astype(int)
    full_train_set_path = config.FULL_TRAIN_SPLIT_PATH
    train_set = util.get_labels_from_dataframe(full_train_set_path)

    # # # # # # # # # # # For debugging Purposes # # # # # # # # # # # # # #
    # t1 = train_set[0][0:24]
    # t2 = train_set[1][0:24]
    # train_set = [t1, t2]
    # # # # # # # # # # # For debugging Purposes # # # # # # # # # # # # # #

    full_train_set = find_files_in_database(complete_database_features,
                                            train_set)
    if mode == 'complete':
        test_set_for_comp = util.get_labels_from_dataframe(config.TEST_SPLIT_PATH)
        full_test_set = find_files_in_database(complete_database_features,
                                               test_set_for_comp)
        create_data_sub_files(full_train_set, data_logger, folds,
                              workspace, full_test_set, mode, gender)
    elif mode == 'sub':
        test_set_for_comp = util.get_labels_from_dataframe(
            config.TEST_SPLIT_PATH, test=True)
        full_test_set = find_files_in_database(complete_database_features,
                                               test_set_for_comp, test=True)
        create_data_sub_files(full_train_set, data_logger, folds, workspace,
                              test_set_for_comp=[full_test_set], mode=mode,
                              gender=gender)
    else:
        raise Exception('An incorrect mode has been detected. Mode can '
                        'either be "complete" or "sub"')
        sys.exit()
