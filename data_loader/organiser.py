import os
import pickle
import torch
import numpy as np
import utilities.utilities_main as util
from data_loader import data_gen
from exp_run import audio_feature_extractor as afe
import pandas as pd
import random
import math


def get_updated_features(data, min_samples, meta, segment_dim, freq_bins,
                         feature_exp, config, label='train'):
    """
    Final stage before segmenting the data into the form N, F, S where N is
    number of new dimensions related to the value of segment_dim. If
    'whole_train' is selected, interpolation will be applied. If the shorten
    dataset option is provided, the files will be shortened before processing

    Inputs:
        data: Array of the data in the database
        min_samples: The size of the shortest file in the database
        label: Set for 'train', 'dev', or 'test'
        meta: Meta data including folder, class, score and original index
        logger: The main logger for recording important information
        segment_dim: The segmentation dimensions for the updated data
        freq_bins: Number of bins for the feature. eg logmel - 64
        amcs: Bool - Audio Mode is Concat not Shorten
        convert_to_im: Bool - will data be converted to 3D?
        initialiser: Used to log information on the first iteration of outer
                     loop
        training: Determines the data processing, 'random_sample',
                  'chunked_file', or 'whole_file'
        max_value: The longest file in the dataset used for 'whole_file'
        feature_exp: Type of features used in this experiment, eg. logmel

    Outputs:
        new_features: Updated array of features N, F, S where S is the
                      feature dimension specified in the config file.
        new_folders: Updated list of folders
        new_classes: Updated list of classes
        new_scores: Updated list of scores
        new_indices: Updated list of indices
    """
    dimension = data.shape[0] // freq_bins
    reshaped_data = np.reshape(data, (freq_bins, dimension))
    # Folder, class, score, gender
    folder, clss, score, gender = meta
    # TODO Change Back
    if label == 'train' and config.EXPERIMENT_DETAILS['CROP']:
        subsample_length = reshaped_data.shape[1] - min_samples
        if subsample_length < 0:
            pass
        else:
            random_pointer = random.randint(0, subsample_length)
            reshaped_data = reshaped_data[:, random_pointer:random_pointer+min_samples]

    # tuple in form of features, folders, classes, scores, genders, indices
    new_meta_data = afe.feature_segmenter(reshaped_data, meta, feature_exp,
                                          segment_dim)
    return new_meta_data


def calculate_length(data, min_samples, label, segment_dim, freq_bins):
    """
    Calculates the length of the updated array once the features have been
    segmented into dimensions specified by segment_dimension in config file

    Inputs:
        data: The data to be segmented
        min_samples: This is the shortest file in the dataset
        label: Set to 'train', 'dev', or 'test'
        segment_dim: Dimensions to reshape the data
        freq_bins: The number of bins used, eg. logmel = 64
        amcs: Bool - Audio Mode is Concat not Shorten

    Output:
        length: The length of the dimension of the data after segmentation
    """
    # TODO Change back
    if not label:
        dimension = data.shape[0] // freq_bins
        if dimension % segment_dim == 0:
            length = (dimension // segment_dim)
        else:
            length = (dimension // segment_dim) + 1
    else:
        if min_samples % segment_dim == 0:
            length = (min_samples // segment_dim)
        else:
            length = (min_samples // segment_dim) + 1

    # #TODO Change back
    # dimension = data.shape[0] // freq_bins
    # if dimension % segment_dim == 0:
    #     length = (dimension // segment_dim)
    # else:
    #     length = (dimension // segment_dim) + 1

    return length


def process_data(freq_bins, features, labels, label, min_samples, logger,
                 segment_dim, feature_exp, config):
    """
    Determine the array size of the dataset once it has been reshaped into
    segments of length equal to that specified in the config file for feature
    dimensions. Following this, create the arrays with respect to the chosen
    feature type of the data. Update the folders, class, score and index lists.

    Inputs:
        amcs: Bool - Audio Mode is Concat not Shorten
        freq_bins: Number of bins for the feature. eg logmel - 64
        features: Array of the features in the database
        labels; Labels corresponding to the features in the database
        label: Set for 'train', 'dev', or 'test'
        min_samples: The size of the shortest file in the database
        logger: The main logger for recording important information
        segment_dim: The segmentation dimensions for the updated data
        convert_to_im: Bool - will data be converted to 3D?
        training: Determines the data processing, 'random_sample',
                  'chunked_file', or 'whole_file'
        max_value: The longest file in the dataset used for 'whole_file'
        feature_exp: Type of features used in this experiment, eg. logmel

    Outputs:
        update_features: Updated array of features N, F, S where S is the
                         feature dimension specified in the config file.
        update_labels: Updated lists containing the folders, classes, scores,
                       and indices after segmentation
        locator: List of the length of every segmented data
    """
    if isinstance(min_samples, list):
        min_samples, indx_to_dict = min_samples
    # TODO Change Back
    # Work out how many dimensions the segmented feature dataset will have
    if label == 'train' and config.EXPERIMENT_DETAILS['CROP']:
        data = features[0, 0]
        length = 0
        for crop_length in min_samples:
            temp_length = calculate_length(data, min_samples[crop_length][0],
                                           True, segment_dim, freq_bins)
            temp_length *= min_samples[crop_length][1]
            length += temp_length
        # length = temp_length * features.shape[0]
        # if config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS'] and \
        #         config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS'] and \
        #         config.EXPERIMENT_DETAILS['CROP']:
        #     length += 30
    else:
        length = 0
        for pointer, i in enumerate(labels[0, :]):
            i = features[pointer, 0]
            temp_length = calculate_length(i, min_samples, False, segment_dim,
                                           freq_bins)
            length += temp_length

    # TODO REMOVE
    # length = 0
    # for pointer, i in enumerate(labels[0, :]):
    #     i = features[pointer, 0]
    #     temp_length = calculate_length(i, min_samples, label, segment_dim,
    #                                    freq_bins)
    #     length += temp_length

    update_features = np.zeros((length, freq_bins, segment_dim),
                               dtype=np.float32)
    pointer = 0
    locator = []
    final_folders = []
    final_classes = []
    final_scores = []
    final_genders = []
    final_indices = []
    initialiser = 0
    tmp = min_samples.copy()
    key_for_double = [i for i in min_samples.keys() if '_2' in i]
    if len(key_for_double) > 0:
        key_for_double = key_for_double[0][:-2]
        indx = indx_to_dict[key_for_double]
        if indx == 0:
            vals = [0, 0]
        elif indx == 1:
            vals = [1, 0]
        elif indx == 2:
            vals = [0, 1]
        else:
            vals = [1, 1]
        locs = [p for p, i in enumerate(labels[1]) if i == vals[0] and
                labels[3][p] == vals[1]]
        sample = min_samples[key_for_double+'_2'][1]
        rnd_sample = random.sample(locs, sample)
    for l, data in enumerate(labels[0, :]):
        data = features[l, 0]
        # Folder, class, score, gender
        meta = [labels[0][l], labels[1][l], labels[2][l], labels[3][l]]

        # if config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS'] and label == \
        #         'train' and config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS']:
        #     if labels[1][i] == 1 and labels[3][i] == 1 and counter < 12:
        #         min_samples = 1210000
        #         counter += 1
        #     elif labels[1][i] == 1 and labels[3][i] == 1 and counter >= 12:
        #         min_samples = 1265000
        #         counter += 1
        #     else:
        #         min_samples = tmp

        if label == 'train' and config.EXPERIMENT_DETAILS['CROP']:
            if labels[1][l] == 0:
                if config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
                    if labels[3][l] == 0:
                        min_samples = tmp['fndep'][0]
                    else:
                        min_samples = tmp['mndep'][0]
                else:
                    min_samples = tmp['ndep'][0]
            else:
                if config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
                    if labels[3][l] == 0:
                        min_samples = tmp['fdep'][0]
                    else:
                        min_samples = tmp['mdep'][0]
                else:
                    min_samples = tmp['dep'][0]
            if len(key_for_double) > 0:
                if l in rnd_sample:
                    min_samples = tmp[key_for_double+'_2'][0]

        (new_features, new_folders, new_classes, new_scores, new_genders,
         new_indices) = get_updated_features(data, min_samples, meta,
                                             segment_dim, freq_bins,
                                             feature_exp, config, label)
        initialiser += 1

        z, _, _ = new_features.shape
        update_features[pointer:pointer + z, :, :] = new_features

        locator.append([pointer, pointer+z])
        new_indices = new_indices + pointer
        final_folders.append(new_folders)
        final_classes.append(new_classes)
        final_scores.append(new_scores)
        final_genders.append(new_genders)
        if type(new_indices) is int:
            final_indices.append(new_indices)
        else:
            final_indices.append(new_indices.tolist())
        pointer += z

    print(f"The dimensions of the {label} features are:"
          f" {update_features.shape}")
    logger.info(f"The dimensions of the {label} features are:"
                f" {update_features.shape}")
    if type(final_folders[0]) is list:
        final_folders = [j for i in final_folders for j in i]
    if type(final_classes[0]) is list:
        final_classes = [j for i in final_classes for j in i]
    if type(final_scores[0]) is list:
        final_scores = [j for i in final_scores for j in i]
    if type(final_genders[0]) is list:
        final_genders = [j for i in final_genders for j in i]
    if type(final_indices[0]) is list:
        final_indices = [j for i in final_indices for j in i]
    update_labels = [final_folders, final_classes, final_scores,
                     final_genders, final_indices]

    return update_features, update_labels, locator


def determine_folds(current_fold, total_folds):
    """
    Function to determine which folds will be used for training and which
    will be held out for validation

    Inputs:
        current_fold: Current fold of the experiment (will be used as
                      validation fold)
        total_folds: How many folds will this dataset have?

    Outputs:
        folds_for_train: List of training folds
        folds_for_dev: List of validation folds
    """
    folds_for_train = []
    for fold_num in list(range(1, total_folds+1)):
        if current_fold == fold_num:
            folds_for_dev = current_fold
        else:
            folds_for_train.append(fold_num)
    return folds_for_train, folds_for_dev


def file_paths(features_dir, config):
    """
    Determines the file paths for the training fold data, validation fold
    data, the summary file created in the data processing stage, and the
    database.

    Inputs:
        features_dir: Directory of the created features
        config: Config file to be used for this experiment
        logger: For logging relevant information
        current_fold: This is used to hold out a specific fold for validation

    Outputs:
        train_meta_file: File paths for the training folds
        dev_meta_file: File path for the validation fold
        sum_file: File path of the summary file (holds meta information)
        h5_database: File path of the database
    """
    train_meta_file = config.TRAIN_SPLIT_PATH
    dev_meta_file = config.DEV_SPLIT_PATH
    sum_file = os.path.join(features_dir, 'summary.pickle')
    if config.GENDER == 'm' or config.GENDER == 'f':
        h5_database = os.path.join(features_dir,
                                   'complete_database'+'_'+config.GENDER+'.h5')
    else:
        h5_database = os.path.join(features_dir, 'complete_database.h5')

    return train_meta_file, dev_meta_file, sum_file, h5_database


def re_distribute(clss):
    max_val = max(len(i) for i in clss.values())
    indices = []

    for i in clss:
        length = len(clss[i]) - 1
        tmp = np.zeros(max_val).astype(int)
        counter = 0
        for j in range(max_val):
            tmp[j] = clss[i][counter]
            counter += 1
            if counter > length:
                counter = 0
        clss[i] = list(tmp)
        indices.append(clss[i])

    indices = [j for i in indices for j in i]

    return clss, indices


def find_weight(zeros, ones, config):
    """
    Finds the balance of the dataset to create weights for training

    Inputs:
        zeros: Dictionary folder: list(indexes)
        ones: Dictionary folder: list(indexes)
        indexes: Dictionary index: folder for every file in dataset

    Output:
        weights: List based on number of specific folders per set (0/1)
        micro_weights: Array - each point's index corresponds to the file
        folder corresponding to indexes[index] = folder. This array contains
        a per file based weight.
    """
    # min_class = min(len(zeros), len(ones))
    # max_class = max(len(zeros), len(ones))
    # class_weight = min_class / max_class
    # index_to_alter = [instance_of_zero, instance_of_one].index(max_class)
    # index_to_alter = [len(zeros), len(ones)].index(max_class)

    zeros_list = [len(i) for i in zeros.values()]
    ones_list = [len(i) for i in ones.values()]
    max_zeros = max(zeros_list)
    min_zeros = min(zeros_list)
    average_zeros = np.average(zeros_list)
    max_ones = max(ones_list)
    min_ones = min(ones_list)
    average_ones = np.average(ones_list)

    # This is a class level weighting
    instance_of_zero = sum(len(i) for i in zeros.values())
    instance_of_one = sum(len(i) for i in ones.values())
    min_class = min(len(zeros), len(ones))
    max_class = max(len(zeros), len(ones))

    class_weight = min_class / max_class
    index_to_alter = [len(zeros), len(ones)].index(max_class)

    weights = [1, 1]
    weights[index_to_alter] = class_weight

    weights_dict = {}
    for i in zeros:
        weights_dict[i] = weights[0]
    for i in ones:
        weights_dict[i] = weights[1]

    # This is a per file representation weight for a given set
    # if instance_of_zero != instance_of_one:
    if config.WEIGHT_TYPE == 'micro' or config.WEIGHT_TYPE == 'both':
        min_zeros = min(len(i) for i in zeros.values())
        min_ones = min(len(i) for i in ones.values())
        micro_weights = {}
        for i in zeros:
            tmp = min_zeros / len(zeros[i])
            if config.WEIGHT_TYPE == 'both':
                tmp = tmp * weights_dict[0]
            micro_weights[i] = tmp
        for i in ones:
            tmp = min_ones / len(ones[i])
            if config.WEIGHT_TYPE == 'both':
                tmp = tmp * weights_dict[1]
            micro_weights[i] = tmp
        weights_dict = micro_weights

    if config.WEIGHT_TYPE == 'instance':
        min_tmp = min(instance_of_zero, instance_of_one)
        max_tmp = max(instance_of_zero, instance_of_one)

        tmp_weight = min_tmp / max_tmp
        index_to_alter = [instance_of_zero, instance_of_one].index(max_tmp)

        weights = [1, 1]
        weights[index_to_alter] = tmp_weight

        instance_weights = {}
        for i in zeros:
            instance_weights[i] = weights[0]
        for i in ones:
            instance_weights[i] = weights[1]
        weights_dict = instance_weights

    return weights_dict, weights


def find_weight2(values, reference):
    """
    Finds the balance of the dataset to create weights for training

    Inputs:
        zeros: Dictionary folder: list(indexes)
        ones: Dictionary folder: list(indexes)
        indexes: Dictionary index: folder for every file in dataset

    Output:
        weights: List based on number of specific folders per set (0/1)
        micro_weights: Array - each point's index corresponds to the file
        folder corresponding to indexes[index] = folder. This array contains
        a per file based weight.
    """
    min_class = min(values, reference)
    max_class = max(values, reference)
    index_to_alter = [values, reference].index(max_class)
    class_weight = min_class / max_class

    weight = [1, 1]
    weight[index_to_alter] = class_weight

    return weight


def data_info(labels, value, logger, config):
    """
    Log the number of ones and zeros in the current set. If class_weights is
    selected, determine the balance of the dataset

    Inputs:
        labels: The labels for the current set of data
        value: Set to training or validation
        logger: To record important information
        config: Config file for state information

    Outputs:
        zeros: Number of zeros in the current set
        zeros_index: Indices of the zeros of the set w.r.t. feature array
        ones: Number of ones in the current set
        ones_index: Indices of the ones of the set w.r.t. feature array
        class_weights: The class weights for current set
    """
    # TODO Remember to evaluate whether this should stay or go
    # zeros, zeros_index, ones, ones_index = util.count_classes(labels[1])

    zeros, zeros_index, ones, ones_index, indices = util.count_classes(labels)
    if config.EXPERIMENT_DETAILS['OVERSAMPLE']:
        zeros, zeros_index = re_distribute(zeros)
        ones, ones_index = re_distribute(ones)

    if config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS'] and value == 'train':
        #TODO Remove this after protoyping
        # num_ones = sum(len(i) for i in ones.values())
        # diff = (len(zeros_index) - num_ones) / 2
        # zeros_index = random.sample(zeros_index, num_ones+int(diff))

        zeros_index = random.sample(zeros_index, sum(len(i) for i in
                                                     ones.values()))
        update_zeros = {}
        for i in zeros_index:
            tmp_folder = indices[i]
            if tmp_folder not in update_zeros:
                update_zeros[tmp_folder] = [i]
            else:
                update_zeros[tmp_folder].append(i)
        # to_delete = []
        # for i in indices:
        #     if i not in zeros_index and i not in ones_index:
        #         to_delete.append(i)
        # for i in to_delete:
        #     del indices[i]
        zeros = update_zeros
        # zeros = util.count_class(labels, update_zeros, zeros, indexes)
        # TODO May need to change back
        # zeros = len(zeros_index)

    print(f"The number of class zero and one files in the {value} split after "
          f"segmentation are {len(zeros_index)}, {len(ones_index)}")
    logger.info(f"The number of class zero files in the {value} split "
                f"after segmentation are {len(zeros_index)}")
    logger.info(f"The number of class one files in the {value} split "
                f"after segmentation are {len(ones_index)}")

    use_class_weights = config.EXPERIMENT_DETAILS['CLASS_WEIGHTS']
    if use_class_weights:
        weights, class_weights = find_weight(zeros, ones, config)
    else:
        class_weights = weights = [1, 1]

    logger.info(f"{config.WEIGHT_TYPE} Weights: {weights}")

    return zeros, zeros_index, ones, ones_index, weights, class_weights


def data_info_gender(labels, value, logger, config):
    """
    Log the number of ones and zeros in the current set. If class_weights is
    selected, determine the balance of the dataset

    Inputs:
        labels: The labels for the current set of data
        value: Set to training or validation
        logger: To record important information
        config: Config file for state information

    Outputs:
        zeros: Number of zeros in the current set
        zeros_index: Indices of the zeros of the set w.r.t. feature array
        ones: Number of ones in the current set
        ones_index: Indices of the ones of the set w.r.t. feature array
        class_weights: The class weights for current set
    """
    zeros, zeros_index, ones, ones_index, indices = util.count_classes_gender(labels)

    #TODO Remove this after protoyping
    # num_ones = sum(len(i) for i in ones.values())
    # diff = (len(zeros_index) - num_ones) / 2
    # zeros_index = random.sample(zeros_index, num_ones+int(diff))

    zeros_f, zeros_m = zeros
    zeros_index_f, zeros_index_m = zeros_index
    ones_f, ones_m = ones
    ones_index_f, ones_index_m = ones_index

    def update(index, ind):
        updates = {}
        for indx in index:
            tmp_folder = ind[indx]
            if tmp_folder not in updates:
                updates[tmp_folder] = [indx]
            else:
                updates[tmp_folder].append(indx)
        return updates

    if value == 'train' and config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS']:
        min_set = min(len(zeros_index_f), len(zeros_index_m),
                      len(ones_index_f), len(ones_index_m))

        zeros_index_f = random.sample(zeros_index_f, min_set)
        zeros_index_m = random.sample(zeros_index_m, min_set)
        ones_index_f = random.sample(ones_index_f, min_set)
        ones_index_m = random.sample(ones_index_m, min_set)

        zeros_f = update(zeros_index_f, indices)
        zeros_m = update(zeros_index_m, indices)
        ones_f = update(ones_index_f, indices)
        ones_m = update(ones_index_m, indices)

        # to_delete = []
        # for i in indices:
        #     if i not in zeros_index_f and i not in zeros_index_m and i not in \
        #             ones_index_f and i not in ones_index_m:
        #         to_delete.append(i)
        # for i in to_delete:
        #     del indices[i]

    # zeros = util.count_class(labels, update_zeros, zeros, indexes)
    # TODO May need to change back
    # zeros = len(zeros_index)

    print(f"The number of class zero and one files in the {value} split after "
          f"segmentation are "
          f"{len(zeros_index_f)+len(zeros_index_m)}, "
          f"{len(ones_index_f)+len(ones_index_m)}")
    print(f"The number of female Non-Depressed: {len(zeros_index_f)}")
    print(f"The number of male Non-Depressed: {len(zeros_index_m)}")
    print(f"The number of female Depressed: {len(ones_index_f)}")
    print(f"The number of male Depressed: {len(ones_index_m)}")

    logger.info(f"The number of class zero files in the {value} split "
                f"after segmentation are {len(zeros_index_f)+len(zeros_index_m)}")
    logger.info(f"The number of class one files in the {value} split "
                f"after segmentation are {len(ones_index_f)+len(ones_index_m)}")
    logger.info(f"The number of female Non-Depressed: {len(zeros_index_f)}")
    logger.info(f"The number of male Non-Depressed: {len(zeros_index_m)}")
    logger.info(f"The number of female Depressed: {len(ones_index_f)}")
    logger.info(f"The number of male Depressed: {len(ones_index_m)}")

    gender_weights, g_weights = gender_split_indices2(zeros_f, ones_f, zeros_m,
                                                      ones_m, config)
    zeros = [zeros_f, zeros_m]
    ones = [ones_f, ones_m]
    zeros_index = [zeros_index_f, zeros_index_m]
    ones_index = [ones_index_f, ones_index_m]
    indices = [zeros_index_f, ones_index_f, zeros_index_m, ones_index_m]

    logger.info(f"{config.WEIGHT_TYPE} Weights: {g_weights}")

    return zeros, zeros_index, ones, ones_index, gender_weights, g_weights


def group_data(features, feat_shape, feat_dim, freq_bins):
    """
    Function to split features into a set determined by feat_dim which will
    be used in batches for training. For example, if features.shape = [10,
    64, 100] and feat_dim = 20, each data will be split into 5 (100/20). As
    there are 10 data, the new dimensions will be 5*10
    Inputs:
        features: Features used for training
        feat_shape: Features.shape
        feat_dim: Dimensions to be used for batches
        freq_bins: Number of bins could me Mel or otherwise
        convert_to_im: Bool, will the features be converted into 3D

    Outputs:
        updated_features: Features split into batch form
        updated_locator: List detailing the length of each data after updating
    """
    if feat_shape[-1] % feat_dim == 0:
        new_dim = (feat_shape[-1] // feat_dim)
    else:
        new_dim = (feat_shape[-1] // feat_dim) + 1
    new_dim2 = new_dim * feat_shape[0]
    updated_features = np.zeros((new_dim2, freq_bins, feat_dim),
                                dtype=np.float32)
    pointer = 0
    updated_locator = []
    for i in features:
        last_dim = i.shape[-1]
        if last_dim % feat_dim == 0:
            leftover = 0
        else:
            leftover = feat_dim - (last_dim % feat_dim)
        i = np.hstack((i, np.zeros((freq_bins, leftover))))
        updated_features[pointer:pointer + new_dim, :, :] = np.split(i,
                                                                     new_dim,
                                                                     axis=1)
        updated_locator.append([pointer, pointer + new_dim])
        pointer += new_dim

    return updated_features, updated_locator


def determine_seconds_segment(seconds_segment, feature_dim, window_size, hop,
                              learning_proc, feat_type):
    """
    Determines the number of samples for a given number of seconds of audio
    data. For example if the sampling rate is 16kHz and the data should be
    clustered to 30s chunks then it will have time dimensionality 16k * 30 =
    480000 samples. However, if the data is in the form of mel bin for
    example, the data has already been processed by a window function to
    compress the data along the time axis and so this must be taken into
    account.

    Inputs:
        seconds_segment: The number of seconds the user wants to cluster
        feature_dim: Number of samples for batching
        window_size: Window size used in cases of logmel for example
        hop: Hop length in cases of logmel for example
        learning_proc: How to process the data, random_sample (each sample
                       length determined by feature_dim), chunked_file (length
                       determined by seconds_segment), or whole_file
        feat_type: What type of audio data are we using? Raw? Logmel?

    Output:
        seconds_segment: Updated in terms of samples rather than seconds
    """
    if learning_proc == 'chunked_file':
        seconds_segment = util.seconds_to_sample(seconds_segment,
                                                 window_size=window_size,
                                                 hop_length=hop,
                                                 feature_type=feat_type)
    elif learning_proc == 'random_sample' or learning_proc == 'whole_file':
        seconds_segment = feature_dim

    return seconds_segment


def gender_split_indices2(fnd, fd, mnd, md, config):
    # total_fnd = len(fnd)
    # total_fd = len(fd)
    # total_mnd = len(mnd)
    # total_md = len(md)
    instance_of_fnd = sum(len(i) for i in fnd.values())
    instance_of_fd = sum(len(i) for i in fd.values())
    instance_of_mnd = sum(len(i) for i in mnd.values())
    instance_of_md = sum(len(i) for i in md.values())

    min_class = min(len(fnd), len(fd), len(mnd), len(md))
    weights = [min_class/len(fnd), min_class/len(fd), min_class/len(mnd),
               min_class/len(md)]

    weights_dict = {}
    for i in fnd:
        weights_dict[i] = weights[0]
    for i in fd:
        weights_dict[i] = weights[1]
    for i in mnd:
        weights_dict[i] = weights[2]
    for i in md:
        weights_dict[i] = weights[3]

    if config.WEIGHT_TYPE == 'instance':
        min_value = min(instance_of_fnd, instance_of_fd, instance_of_mnd,
                        instance_of_md)

        weights = [min_value / instance_of_fnd, min_value / instance_of_fd,
                   min_value / instance_of_mnd, min_value / instance_of_md]

        weights_dict = {}
        for i in fnd:
            weights_dict[i] = weights[0]
        for i in fd:
            weights_dict[i] = weights[1]
        for i in mnd:
            weights_dict[i] = weights[2]
        for i in md:
            weights_dict[i] = weights[3]

    return weights_dict, weights


def gender_split_indices(label_data):
    if isinstance(label_data, list):
        label_data = np.array(label_data)
    male_dep_indices = list(np.where((label_data[1, :] == 1) & (label_data[-2, :] == 1))[0])
    male_ndep_indices = list(np.where((label_data[1, :] == 0) & (label_data[-2, :] ==
                                                           1))[0])
    female_dep_indices = list(np.where((label_data[1, :] == 1) & (label_data[-2, :] ==
                                                            0))[0])
    female_ndep_indices = list(np.where((label_data[1, :] == 0) & (label_data[-2, :] ==
                                                             0))[0])

    male_dep = len(male_dep_indices)
    male_ndep = len(male_ndep_indices)
    fem_dep = len(female_dep_indices)
    fem_ndep = len(female_ndep_indices)

    min_value = min([male_dep, male_ndep, fem_dep, fem_ndep])

    fem_nd_w = find_weight2(fem_ndep, min_value)
    fem_d_w = find_weight2(fem_dep, min_value)
    male_nd_w = find_weight2(male_ndep, min_value)
    male_d_w = find_weight2(male_dep, min_value)
    weights = (torch.Tensor(fem_nd_w), torch.Tensor(fem_d_w),
               torch.Tensor(male_nd_w), torch.Tensor(male_d_w))

    weights_per_index = {}
    for i in female_ndep_indices:
        weights_per_index[i] = fem_nd_w[0]
    for i in female_dep_indices:
        weights_per_index[i] = fem_d_w[0]
    for i in male_ndep_indices:
        weights_per_index[i] = male_nd_w[0]
    for i in male_dep_indices:
        weights_per_index[i] = male_d_w[0]

    return female_ndep_indices, female_dep_indices, male_ndep_indices, \
           male_dep_indices, weights_per_index, weights


def get_lengths(data, labels, config, gender=False):
    exp_type = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    feature_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    freq_bins = config.EXPERIMENT_DETAILS['FREQ_BINS']
    lengths_dict = {}
    segments_dict = {}
    for p in range(len(labels[0])):
        d = data[p][0]
        if exp_type == 'mel' or exp_type == 'logmel':
            d = d.reshape(freq_bins, -1)
        if d.shape[-1] % feature_dim == 0:
            seg = d.shape[-1] // feature_dim
        else:
            seg = (d.shape[-1] // feature_dim) + 1

        if labels[1][p] == 0:
            if 'ndep' not in lengths_dict:
                lengths_dict['ndep'] = [d.shape[-1]]
                segments_dict['ndep'] = [seg]
            else:
                lengths_dict['ndep'].append(d.shape[-1])
                segments_dict['ndep'].append(seg)
        else:
            if 'dep' not in lengths_dict:
                lengths_dict['dep'] = [d.shape[-1]]
                segments_dict['dep'] = [seg]
            else:
                lengths_dict['dep'].append(d.shape[-1])
                segments_dict['dep'].append(seg)

        if gender:
            if labels[1][p] == 0 and labels[3][p] == 0:
                if 'fndep' not in lengths_dict:
                    lengths_dict['fndep'] = [d.shape[-1]]
                    segments_dict['fndep'] = [seg]
                else:
                    lengths_dict['fndep'].append(d.shape[-1])
                    segments_dict['fndep'].append(seg)
            elif labels[1][p] == 0 and labels[3][p] == 1:
                if 'mndep' not in lengths_dict:
                    lengths_dict['mndep'] = [d.shape[-1]]
                    segments_dict['mndep'] = [seg]
                else:
                    lengths_dict['mndep'].append(d.shape[-1])
                    segments_dict['mndep'].append(seg)
            elif labels[1][p] == 1 and labels[3][p] == 0:
                if 'fdep' not in lengths_dict:
                    lengths_dict['fdep'] = [d.shape[-1]]
                    segments_dict['fdep'] = [seg]
                else:
                    lengths_dict['fdep'].append(d.shape[-1])
                    segments_dict['fdep'].append(seg)
            else:
                if 'mdep' not in lengths_dict:
                    lengths_dict['mdep'] = [d.shape[-1]]
                    segments_dict['mdep'] = [seg]
                else:
                    lengths_dict['mdep'].append(d.shape[-1])
                    segments_dict['mdep'].append(seg)

    return lengths_dict, segments_dict


def crop_sections(lengths, segments, config, gender=False):
    exp_type = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    feature_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    freq_bins = config.EXPERIMENT_DETAILS['FREQ_BINS']
    min_crop = config.MIN_CROP

    ndep_crop = min(lengths['ndep'])
    ndep_seg = min(segments['ndep'])
    dep_crop = min(lengths['dep'])
    dep_seg = min(segments['dep'])

    segs = [ndep_seg, dep_seg]
    num_ndep = len(lengths['ndep'])
    num_dep = len(lengths['dep'])

    # If using the shortest file in whole dataset, make all set mins =
    # dataset min
    if min_crop:
        min_crop_val = min(ndep_crop, dep_crop)
        min_seg_val = min(ndep_seg, dep_seg)
        ndep_seg = min_seg_val
        dep_seg = min_seg_val
        ndep_crop = min_crop_val
        dep_crop = min_crop_val

    final_ndep = num_ndep * ndep_seg
    final_dep = num_dep * dep_seg

    crops = {'ndep': [ndep_crop, num_ndep], 'dep': [dep_crop, num_dep]}
    dict_to_indx = {0: 'ndep', 1: 'dep'}
    indx_to_dict = {'ndep': 0, 'dep': 1}

    minimum_value = min(final_ndep, final_dep)

    if gender:
        fndep_crop = min(lengths['fndep'])
        mndep_crop = min(lengths['mndep'])
        fdep_crop = min(lengths['fdep'])
        mdep_crop = min(lengths['mdep'])

        fndep_seg = min(segments['fndep'])
        mndep_seg = min(segments['mndep'])
        fdep_seg = min(segments['fdep'])
        mdep_seg = min(segments['mdep'])
        segs = [fndep_seg, fdep_seg, mndep_seg, mdep_seg]

        # If using the shortest file in whole dataset, make all set mins =
        # dataset min
        if min_crop:
            min_crop_val = min(fndep_crop, mndep_crop, fdep_crop, mdep_crop)
            min_seg_val = min(fndep_seg, mndep_seg, fdep_seg, mdep_seg)
            fndep_seg = min_seg_val
            mndep_seg = min_seg_val
            fdep_seg = min_seg_val
            mdep_seg = min_seg_val
            fndep_crop = min_crop_val
            mndep_crop = min_crop_val
            fdep_crop = min_crop_val
            mdep_crop = min_crop_val


        num_fndep = len(lengths['fndep'])
        num_fdep = len(lengths['fdep'])
        num_mndep = len(lengths['mndep'])
        num_mdep = len(lengths['mdep'])
        nums = [num_fndep, num_fdep, num_mndep, num_mdep]
        crops = {'fndep': [fndep_crop, num_fndep], 'fdep': [fdep_crop,
                                                            num_fdep],
                 'mndep': [mndep_crop, num_mndep], 'mdep': [mdep_crop,
                                                            num_mdep]}

        final_fndep = num_fndep * fndep_seg
        final_fdep = num_fdep * fdep_seg
        final_mndep = num_mndep * mndep_seg
        final_mdep = num_mdep * mdep_seg

        finals = [final_fndep, final_fdep, final_mndep, final_mdep]
        dict_to_indx = {0: 'fndep', 1: 'fdep', 2: 'mndep', 3: 'mdep'}
        indx_to_dict = {'fndep': 0, 'fdep': 1, 'mndep': 2, 'mdep': 3}

        minimum_value_gen = min(finals)
        half_min = minimum_value // 2
        if half_min > minimum_value_gen:
            indx = finals.index(minimum_value_gen)
            tmp = nums[indx] * segs[indx]
            if tmp > half_min:
                tmp = math.ceil(half_min / nums[indx])
                tmp_total = tmp * nums[indx]
                if tmp_total > half_min:
                    difference = tmp_total - half_min
                    # if exp_type == 'mel' or exp_type == 'logmel':
                    additions1 = tmp * feature_dim
                    additions2 = (tmp-1) * feature_dim
                    temp = crops[dict_to_indx[indx]]
                    temp[1] -= difference
                    crops[dict_to_indx[indx]] = [int(additions1), temp[1]]
                    replacement = dict_to_indx[indx] + '_2'
                    crops[replacement] = [additions2, difference]
            else:
                difference = half_min - minimum_value_gen
                additions = difference + minimum_value_gen
                index = finals.index(minimum_value_gen)
                additions -= segs[index]
                additions /= (nums[index] - 1)
                segs[index] = int(additions)

                # if exp_type == 'mel' or exp_type == 'logmel':
                additions = additions * feature_dim
                temp = crops[dict_to_indx[index]]
                temp[1] -= 1
                crops[dict_to_indx[index]] = [int(additions), temp[1]]
                replacement = dict_to_indx[index] +'_2'
                crops[replacement] = [temp[0], 1]

    return [crops, indx_to_dict]


def determine_crops(features, labels, config, gender, logger, mode_label):
    use_crop = config.EXPERIMENT_DETAILS['CROP']
    minimise_crop = config.MIN_CROP
    sub_sample = config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS']

    lengths, segments = get_lengths(features, labels, config, gender)
    min_samples, indx_to_dict = crop_sections(lengths, segments, config, gender)
    if mode_label == 'train':
        logger.info(f"Crop Partitions: {min_samples}")

    def get_min(samples):
        min_val = 1e12
        for s in samples:
            if samples[s][0] < min_val:
                min_val = int(min_samples[s][0])
        return min_val

    def set_min(samples, min_val):
        for s in samples:
            samples[s][0] = min_val
        return samples

    if not sub_sample and use_crop and gender and not minimise_crop:
        ndep_min = min(min_samples['fndep'][0], min_samples['mndep'][0])
        dep_min = min(min_samples['fdep'][0], min_samples['mdep'][0],
                      min_samples['mdep_2'][0])
        min_samples['fndep'][0] = ndep_min
        min_samples['mndep'][0] = ndep_min
        min_samples['fdep'][0] = dep_min
        min_samples['mdep'][0] = dep_min
        min_samples['mdep'][1] += 1
        del min_samples['mdep_2']
    elif sub_sample and use_crop and gender and minimise_crop:
        pass
    elif minimise_crop and use_crop:
        min_val = get_min(min_samples)
        min_samples = set_min(min_samples, min_val)

    return [min_samples, indx_to_dict]


def organise_data(config, logger, labels, database, min_samples,
                  list_of_samples, mode_label='train'):
    freq_bins = config.EXPERIMENT_DETAILS['FREQ_BINS']
    feature_exp = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    feature_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    gender = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    features = util.load_data(database, labels)

    min_samples = determine_crops(features, labels, config, gender,
                                  logger, mode_label)

    features, labels, loc = process_data(freq_bins, features, labels,
                                         mode_label, min_samples, logger,
                                         feature_dim, feature_exp, config)

    if gender and config.EXPERIMENT_DETAILS['CROP']:
        zeros, zero_index, ones, one_index, weights, set_weights = \
            data_info_gender(labels, mode_label, logger, config)
    else:
        zeros, zero_index, ones, one_index, weights, set_weights = data_info(
            labels, mode_label, logger, config)

    index = [zero_index, one_index]

    return features, labels, index, loc, (zeros, ones, weights, set_weights)


def log_train_data(logger, class_data, gender_weights=None, train_index=None):
    logger.info(f"\nThe per class class weights (Non-Depresed vs Depressed) "
                f"are: {class_data}")
    if gender_weights is not None:
        female_tot = len(train_index[0]) + len(train_index[1])
        male_tot = len(train_index[2]) + len(train_index[3])
        nd_tot = len(train_index[0]) + len(train_index[2])
        d_tot = len(train_index[1]) + len(train_index[3])
        comp = female_tot + male_tot
        table = np.array([[len(train_index[0]), len(train_index[1]), female_tot],
                          [len(train_index[2]), len(train_index[3]),
                           male_tot], [nd_tot, d_tot, comp]])
        p = pd.DataFrame(data=table, index=['Female', 'Male', 'Total'],
                         columns=['Non-Dep', 'Dep', 'Total'])
        logger.info(f"\n{p}\n")
        logger.info(f"\nThe Gender Weights are: \nFemale_Non_Dep: "
                    f"{gender_weights[0]}\nFemale_Dep: "
                    f"{gender_weights[1]}\nMale_Non_Dep: "
                    f"{gender_weights[2]}\nMale_Dep: "
                    f"{gender_weights[3]}\n")


def run(config, logger, checkpoint, features_dir, data_saver):
    """
    High level function to process the training and validation data. This
    function obtains the file locations, folds for training/validation sets,
    processes the data to be used for training.
    Inputs:
        config: Config file holding state information for experiment
        logger: Logger to record important information
        current_fold: Current fold for experiment to be used to determine the
                      training and validation folds
        checkpoint: Is there a checkpoint to load from?

    Outputs:
        generator: Generator for training and validation batch data loading
        class_weights_train:
        class_weights_dev
        zero_train
        one_train
    """
    train_file, dev_file, summary_file, database = file_paths(features_dir,
                                                              config)
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    logger.info(f"The dimensions of the logmel features before segmentation "
                f"are: {summary[1][-1]}")
    if config.EXPERIMENT_DETAILS['FEATURE_EXP'] == 'raw':
        min_samples = int(summary[1][summary[0].index('MinSamples')])
    else:
        min_samples = int(summary[1][summary[0].index('MinWindows')])

    list_of_samples = summary[1][summary[0].index('ListOfSamples')]

    data1 = util.csv_read(train_file)
    train_labels = np.array([[i[0] for i in data1], [i[1] for i in data1],
                            [i[2] for i in data1],
                            [i[3] for i in data1]]).astype(int)

    data2 = util.csv_read(dev_file)
    dev_labels = np.array([[i[0] for i in data2], [i[1] for i in data2],
                          [i[2] for i in data2],
                          [i[3] for i in data2]]).astype(int)

    path = config.TEST_SPLIT_PATH
    data3 = util.csv_read(path)
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

    # class data is tuple (number zeros, number ones, class weights)
    train_features, train_labels, train_index, train_loc, class_data = \
        organise_data(config, logger, train_labels, database, min_samples,
                      list_of_samples, mode_label='train')

    dev_features, dev_labels, dev_index, dev_loc, _ = organise_data(config,
                                                                    logger,
                                                                    dev_labels,
                                                                    database,
                                                                    min_samples,
                                                                    list_of_samples,
                                                                    mode_label='dev')
    gender_balance = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    if gender_balance:
        # Gender weights tuple of (fem_nd_w, fem_d_w, male_nd_w, male_d_w)
        # TODO May have to fix this
        # female_ndep_ind, female_dep_ind, male_ndep_ind, male_dep_ind, \
        # gender_weights = gender_split_indices(train_labels)
        # zeros, ones, class_weights = class_data
        female_ndep_ind = train_index[0][0]
        male_ndep_ind = train_index[0][1]
        female_dep_ind = train_index[1][0]
        male_dep_ind = train_index[1][1]
        # per instance weights
        # female_ndep_ind, female_dep_ind, male_ndep_ind, male_dep_ind, \
        # gender_weights, g_weights = gender_split_indices(train_labels)
        # per unique instance weights
        # gender_weights, g_weights = gender_split_indices2(female_ndep_ind,
        #                                                   male_ndep_ind,
        #                                                   female_dep_ind,
        #                                                   male_dep_ind)
        train_index = [female_ndep_ind, female_dep_ind, male_ndep_ind,
                       male_dep_ind]
        log_train_data(logger, class_data[-2], class_data[-1], train_index)

        # class_data = (zeros, ones, gender_weights)

        dev_index = [dev_index[0][0], dev_index[0][1], dev_index[1][0],
                     dev_index[1][1]]

        for p, i in enumerate(dev_labels[3]):
            if dev_labels[1][p] == 0 and i == 0:
                pass
            elif dev_labels[1][p] == 0 and i == 1:
                dev_labels[1][p] = 2
            elif dev_labels[1][p] == 1 and i == 0:
                dev_labels[1][p] = 1
            else:
                dev_labels[1][p] = 3
    else:
        log_train_data(logger, class_data[-1])

    zeros, ones, weights, set_weights = class_data
    dev_weights = {}
    for i in range(len(dev_labels[0])):
        f = dev_labels[0][i]
        clss = dev_labels[1][i]
        gender = dev_labels[3][i]
        if clss == 0 or clss == 2:
            if gender_balance and gender == 0:
                dev_weights[f] = set_weights[0]
            elif gender_balance and gender == 1:
                dev_weights[f] = set_weights[2]
            else:
                dev_weights[f] = set_weights[0]
        else:
            if gender_balance and gender == 0:
                dev_weights[f] = set_weights[1]
            elif gender_balance and gender == 1:
                dev_weights[f] = set_weights[3]
            else:
                dev_weights[f] = set_weights[1]
    class_data = zeros, ones, weights, set_weights, dev_weights

    generator = data_gen.GenerateData(train_labels=train_labels,
                                      dev_labels=dev_labels,
                                      train_feat=train_features,
                                      dev_feat=dev_features,
                                      train_loc=train_loc,
                                      dev_loc=dev_loc,
                                      train_indices=train_index,
                                      dev_indices=dev_index,
                                      logger=logger,
                                      config=config,
                                      checkpoint=checkpoint,
                                      gender_balance=gender_balance,
                                      data_saver=data_saver)

    return generator, class_data
