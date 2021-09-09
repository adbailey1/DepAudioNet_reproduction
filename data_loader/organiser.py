import os
import pickle
import numpy as np
import utilities.utilities_main as util
from data_loader import data_gen
from exp_run import audio_feature_extractor as afe
import pandas as pd
import random


def per_gender_indices(label_data):
    if isinstance(label_data, list):
        label_data = np.array(label_data)
    male_dep_indices = list(np.where((label_data[1, :] == 1) & (label_data[-2, :] == 1))[0])
    male_ndep_indices = list(np.where((label_data[1, :] == 0) & (label_data[-2, :] ==
                                                           1))[0])
    female_dep_indices = list(np.where((label_data[1, :] == 1) & (label_data[-2, :] ==
                                                            0))[0])
    female_ndep_indices = list(np.where((label_data[1, :] == 0) & (label_data[-2, :] ==
                                                             0))[0])

    return female_ndep_indices, female_dep_indices, male_ndep_indices, \
           male_dep_indices


def get_updated_features(data, min_samples, meta, segment_dim, freq_bins,
                         feature_exp, config, mode_label='train'):
    """
    Final stage before segmenting the data into the form N, F, S where N is
    number of new dimensions related to the value of segment_dim. If
    'whole_train' is selected, interpolation will be applied. If the shorten
    dataset option is provided, the files will be shortened before processing

    Inputs:
        data: Array of the data in the database
        min_samples: The size of the shortest file in the database
        meta: Meta data including folder, class, score and gender
        segment_dim: The segmentation dimensions for the updated data
        freq_bins: Number of bins for the feature. eg logmel - 64
        feature_exp: Type of features used in this experiment, eg. logmel
        config: config file holding state information for experiment
        mode_label: Set for 'train', 'dev', or 'test'

    Outputs:
        new_meta_data: updated array of features (N, F, S where S is the
                      feature dimension specified in the config file),
                      updated list of folders, classes, scores, genders,
                      and indices
    """
    dimension = data.shape[0] // freq_bins
    reshaped_data = np.reshape(data, (freq_bins, dimension))
    if mode_label == 'train' and config.EXPERIMENT_DETAILS['CROP']:
        subsample_length = reshaped_data.shape[1] - min_samples
        if subsample_length < 0:
            pass
        else:
            random_pointer = random.randint(0, subsample_length)
            reshaped_data = reshaped_data[:,
                            random_pointer:random_pointer+min_samples]

    # tuple in form of features, folders, classes, scores, genders, indices
    new_meta_data = afe.feature_segmenter(reshaped_data,
                                          meta,
                                          feature_exp,
                                          segment_dim)
    return new_meta_data


def calculate_length(data, min_samples, use_min_sample, segment_dim, freq_bins):
    """
    Calculates the length of the updated array once the features have been
    segmented into dimensions specified by segment_dimension in config file

    Inputs:
        data: The data to be segmented
        min_samples: This is the shortest file in the dataset
        use_min_sample: bool to use already calculated segment length
                        according to the minimum of the data
        segment_dim: Dimensions to reshape the data
        freq_bins: The number of bins used, eg. logmel = 64

    Output:
        length: The length of the dimension of the data after segmentation
    """
    if not use_min_sample:
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

    return length


def process_data(freq_bins, features, labels, mode_label, min_samples, logger,
                 segment_dim, feature_exp, config):
    """
    Determine the array size of the dataset once it has been reshaped into
    segments of length equal to that specified in the config file for feature
    dimensions. Following this, create the arrays with respect to the chosen
    feature type of the data. Update the folders, class, score and index lists.

    Inputs:
        freq_bins: Number of bins for the feature. eg logmel - 64
        features: Array of the features in the database
        labels; Labels corresponding to the features in the database
        mode_label: Set for 'train', 'dev', or 'test'
        min_samples: The size of the shortest file in the database
        logger: The main logger for recording important information
        segment_dim: The segmentation dimensions for the updated data
        feature_exp: Type of features used in this experiment, eg. logmel
        config: config file holding state information for experiment


    Outputs:
        update_features: Updated array of features N, F, S where S is the
                         feature dimension specified in the config file.
        update_labels: Updated lists containing the folders, classes, scores,
                       and indices after segmentation
        locator: List of the length of every segmented data
    """
    if isinstance(min_samples, list):
        min_samples, indx_to_dict = min_samples
    # Work out how many dimensions the segmented feature dataset will have
    if mode_label == 'train' and config.EXPERIMENT_DETAILS['CROP']:
        data = features[0, 0]
        length = 0
        for crop_length in min_samples:
            temp_length = calculate_length(data,
                                           min_samples[crop_length][0],
                                           True,
                                           segment_dim,
                                           freq_bins)
            temp_length *= min_samples[crop_length][1]
            length += temp_length
    else:
        length = 0
        for pointer, i in enumerate(labels[0, :]):
            i = features[pointer, 0]
            temp_length = calculate_length(i,
                                           min_samples,
                                           False,
                                           segment_dim,
                                           freq_bins)
            length += temp_length

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
    # Convert the updated labels (0, 1, 2, 3) back to class and gender
    # e.g. 0 == [fem, ndep], 1 == [fem, dep] etc.
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
    for lab, data in enumerate(labels[0, :]):
        data = features[lab, 0]
        # Folder, class, score, gender
        meta = [labels[0][lab], labels[1][lab], labels[2][lab], labels[3][lab]]

        if mode_label == 'train' and config.EXPERIMENT_DETAILS['CROP']:
            if labels[1][lab] == 0:
                if labels[3][lab] == 0:
                    min_samples = tmp['fndep'][0]
                else:
                    min_samples = tmp['mndep'][0]
            else:
                if labels[3][lab] == 0:
                    min_samples = tmp['fdep'][0]
                else:
                    min_samples = tmp['mdep'][0]
            if len(key_for_double) > 0:
                if lab in rnd_sample:
                    min_samples = tmp[key_for_double+'_2'][0]

        (new_features, new_folders, new_classes, new_scores, new_genders,
         new_indices) = get_updated_features(data,
                                             min_samples,
                                             meta,
                                             segment_dim,
                                             freq_bins,
                                             feature_exp,
                                             config,
                                             mode_label)
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

    print(f"The dimensions of the {mode_label} features are:"
          f" {update_features.shape}")
    logger.info(f"The dimensions of the {mode_label} features are:"
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


def file_paths(features_dir, config):
    """
    Determines the file paths for the training fold data, validation fold
    data, the summary file created in the data processing stage, and the
    database.

    Inputs:
        features_dir: Directory of the created features
        config: Config file to be used for this experiment

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
    """
    Finds the largest file from the data and iterates through every
    subsequent file to make the length of indices match the maximum value. It
    does this by simply iterating through the existing indices and adds
    them to an updated list until the counter reaches the end, and then the
    counter resets, adding the indices again from the beginning to the new
    updated list - thus oversampling.
    Inputs:
        clss: Dictionary Key is folder, Value is List(indices)

    Outputs:
        clss: Dictionary Key is folder, Value is updated List(indices)
        indices: Complete list of updated indices
    """
    # Determine largest folder w.r.t. indices
    max_val = max(len(list_inx) for list_inx in clss.values())
    indices = []

    for folder in clss:
        length = len(clss[folder]) - 1
        tmp = np.zeros(max_val).astype(int)
        counter = 0
        for i in range(max_val):
            tmp[i] = clss[folder][counter]
            counter += 1
            if counter > length:
                counter = 0
        clss[folder] = list(tmp)
        indices.append(clss[folder])

    indices = [j for i in indices for j in i]

    return clss, indices


def find_weight(zeros, ones, config):
    """
    Finds the balance of the dataset to create weights for training

    Inputs:
        zeros: Dictionary Key is folder, Value is list(indexes)
        ones: Dictionary Key is folder, Value is list(indexes)
        indexes: Dictionary Key is index, Value is folder

    Output:
        weights_dict: Dictionary Key is folder, Value is the respective weight
        weights: List of the weights for each partition ndep, dep
    """
    # This is a class level weighting or "macro" weighting - based on the
    # total number of interview files for each set
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

    # Micro is calculated by finding the minimum interview (in terms of
    # segments) for each set (e.g. File 0 has min for depressed with 10
    # segments). "both" combines macro and micro
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

    # "instance" weight calculates the weight based on the total number of
    # segments for a given set (non-dep vs dep)
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


def data_info(labels, mode_label, logger, config, hidden_test=False):
    """
    Log the number of ones and zeros in the current set. If class_weights is
    selected, determine the balance of the dataset

    Inputs:
        labels: The labels for the current set of data
        mode_label: Set to training or validation
        logger: To record important information
        config: Config file for state information

    Outputs:
        zeros: Dictionary, Key is folder, Value are the indices
        zeros_index: List indices of the zeros of the set w.r.t. feature array
        ones: Dictionary, Key is folder, Value are the indices
        ones_index: List indices of the ones of the set w.r.t. feature array
        weights: Dictionary, Key is folder, Value is respective weight
        class_weights: List of the class weights for current set in form [
        ndep, dep]
    """

    zeros, zeros_index, ones, ones_index, indices = util.count_classes_gender(labels)

    zeros_index_f, zeros_index_m = zeros_index
    ones_index_f, ones_index_m = ones_index
    if mode_label == 'train' and config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS']:
        min_set = min(len(zeros_index_f), len(zeros_index_m),
                      len(ones_index_f), len(ones_index_m))
        min_set *= 2

    zeros, zeros_index, ones, ones_index, indices = util.count_classes(labels)

    if config.EXPERIMENT_DETAILS['OVERSAMPLE']:
        zeros, zeros_index = re_distribute(zeros)
        ones, ones_index = re_distribute(ones)

    if mode_label == 'train' and config.EXPERIMENT_DETAILS[
       'SUB_SAMPLE_ND_CLASS']:
        zeros_index = random.sample(zeros_index, min_set)
        ones_index = random.sample(ones_index, min_set)
        update_zeros = {}
        for i in zeros_index:
            tmp_folder = indices[i]
            if tmp_folder not in update_zeros:
                update_zeros[tmp_folder] = [i]
            else:
                update_zeros[tmp_folder].append(i)
        zeros = update_zeros
        update_ones = {}
        for i in ones_index:
            tmp_folder = indices[i]
            if tmp_folder not in update_ones:
                update_ones[tmp_folder] = [i]
            else:
                update_ones[tmp_folder].append(i)
        ones = update_ones

    print(f"The number of class zero and one files in the {mode_label} "
          f"split after segmentation are {len(zeros_index)}, {len(ones_index)}")
    logger.info(f"The number of class zero files in the {mode_label} split "
                f"after segmentation are {len(zeros_index)}")
    logger.info(f"The number of class one files in the {mode_label} split "
                f"after segmentation are {len(ones_index)}")

    use_class_weights = config.EXPERIMENT_DETAILS['CLASS_WEIGHTS']
    if use_class_weights:
        weights, class_weights = find_weight(zeros,
                                             ones,
                                             config)
    else:
        class_weights = weights = [1, 1]

    logger.info(f"{config.WEIGHT_TYPE} Weights: {weights}")

    return zeros, zeros_index, ones, ones_index, weights, class_weights


def data_info_gender(labels, mode_label, logger, config, hidden_test=False):
    """
    Log the number of ones and zeros in the current set. If class_weights is
    selected, determine the balance of the dataset

    Inputs:
        labels: The labels for the current set of data
        mode_label: Set to training or validation
        logger: To record important information
        config: Config file for state information

    Outputs:
        zeros: List (fem, male) of dictionary, Key is folder, Value are the
               indices
        zeros_index: List (fem, male) of list indices of the zeros of the set
                     w.r.t. feature array
        ones: List (fem, male) of dictionary, Key is folder, Value are the
              indices
        ones_index: List (fem, male) of list indices of the ones of the set
                    w.r.t. feature array
        gender_weights: Dictionary, Key is folder, Value is respective weight
        g_weights: List of the class weights for data partition fndep, fdep,
                   mndep, mdep
    """
    zeros, zeros_index, ones, ones_index, indices = util.count_classes_gender(labels)

    zeros_f, zeros_m = zeros
    zeros_index_f, zeros_index_m = zeros_index
    ones_f, ones_m = ones
    ones_index_f, ones_index_m = ones_index

    def update(list_of_indices, index_to_folder):
        updates = {}
        for indx in list_of_indices:
            tmp_folder = index_to_folder[indx]
            if tmp_folder not in updates:
                updates[tmp_folder] = [indx]
            else:
                updates[tmp_folder].append(indx)
        return updates

    if mode_label == 'train' and config.EXPERIMENT_DETAILS[
       'SUB_SAMPLE_ND_CLASS']:
        min_set = min(len(zeros_index_f), len(zeros_index_m),
                      len(ones_index_f), len(ones_index_m))

        zeros_index_f = random.sample(zeros_index_f, min_set)
        zeros_index_m = random.sample(zeros_index_m, min_set)
        ones_index_f = random.sample(ones_index_f, min_set)
        ones_index_m = random.sample(ones_index_m, min_set)

        zeros_f = update(zeros_index_f,
                         indices)
        zeros_m = update(zeros_index_m,
                         indices)
        ones_f = update(ones_index_f,
                        indices)
        ones_m = update(ones_index_m,
                        indices)

    print(f"The number of class zero and one files in the {mode_label} split "
          f"after "
          f"segmentation are "
          f"{len(zeros_index_f)+len(zeros_index_m)}, "
          f"{len(ones_index_f)+len(ones_index_m)}")
    print(f"The number of female Non-Depressed: {len(zeros_index_f)}")
    print(f"The number of male Non-Depressed: {len(zeros_index_m)}")
    print(f"The number of female Depressed: {len(ones_index_f)}")
    print(f"The number of male Depressed: {len(ones_index_m)}")

    logger.info(f"The number of class zero files in the {mode_label} split "
                f"after segmentation are "
                f"{len(zeros_index_f)+len(zeros_index_m)}")
    logger.info(f"The number of class one files in the {mode_label} split "
                f"after segmentation are {len(ones_index_f)+len(ones_index_m)}")
    logger.info(f"The number of female Non-Depressed: {len(zeros_index_f)}")
    logger.info(f"The number of male Non-Depressed: {len(zeros_index_m)}")
    logger.info(f"The number of female Depressed: {len(ones_index_f)}")
    logger.info(f"The number of male Depressed: {len(ones_index_m)}")

    if not hidden_test:
        gender_weights, g_weights = gender_split_indices(zeros_f,
                                                         ones_f,
                                                         zeros_m,
                                                         ones_m,
                                                         config)
    else:
        gender_weights = {i: 1 for i in labels[0]}
        g_weights = [1, 1, 1, 1]
    zeros = [zeros_f, zeros_m]
    ones = [ones_f, ones_m]
    zeros_index = [zeros_index_f, zeros_index_m]
    ones_index = [ones_index_f, ones_index_m]

    logger.info(f"{config.WEIGHT_TYPE} Weights: {g_weights}")

    return zeros, zeros_index, ones, ones_index, gender_weights, g_weights


def gender_split_indices(fnd, fd, mnd, md, config):
    """
    Used to calculate the weights for the respective data partitions:
    Female-Non_Dep, Female-Dep, Male-Non_Dep, Male-Dep

    Inputs:
        fnd: Dictionary of female non_dep, Key is folder, Value is list(indices)
        fd: Dictionary of female dep, Key is folder, Value is list(indices)
        mnd: Dictionary of male non_dep, Key is folder, Value is list(indices)
        md: Dictionary of male dep, Key is folder, Value is list(indices)
        config: Config file holding state information for experiment

    Outputs:
        weights_dict: Dictionary Key is folder, Value is the respective weight
        weights: List of the weights for each partition fndep, fdep, mndep, mdep
    """
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


def get_lengths(data, labels, config):
    """
    Finds the size of every file w.r.t. data subsets 'fndep', 'fdep',
    'mndep', or 'mdep' according to value chosen in config file:
    config.EXPERIMENT_DETAILS['FREQ_BINS'], and also finds the number of
    segments each file will be split into according to the value chosen in
    config file: config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']: 61440
    Inputs:
        data: Array of the feature data for the experiment
        labels: holds the folder, class, score, gender, index
        config: config file holding state information for experiment

    Outputs:
        lengths_dict: Dictionary, Key is 'fndep' / 'fdep' / 'mndep' / 'mdep',
                      Value is list of raw lengths of every file w.r.t. key
        segments_dict: Dictionary, Key is 'fndep' / 'fdep' / 'mndep' / 'mdep',
                       Value is list of number of segments for each file
    """
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


def crop_sections(lengths, config):
    """
    Uses the raw lengths of each subset of the data calculated from
    get_lengths(data, labels, config) to calculate the minimum sample for
    each subset of the data and how many files are present for each subset of
    the data. Subsets include: 'fndep', 'fdep', 'mndep', 'mdep'
    Inputs:
        lengths: Dictionary, Key is 'fndep' / 'fdep' / 'mndep' / 'mdep',
                 Value is list of raw lengths of every file w.r.t. key
        config: config file holding state information for experiment

    Outputs:
        crops: Dictionary, Key is 'fndep' / 'fdep' / 'mndep' / 'mdep',
               Value is list containing the minimum sample length and
               the number of files relating to the key
        indx_to_dict: Dictionary Key is same as min_samples, value is index 0-3
    """
    min_crop = config.MIN_CROP

    fndep_crop = min(lengths['fndep'])
    mndep_crop = min(lengths['mndep'])
    fdep_crop = min(lengths['fdep'])
    mdep_crop = min(lengths['mdep'])

    # If using the shortest file in whole dataset, make all set mins =
    # dataset min
    if min_crop:
        min_crop_val = min(fndep_crop, mndep_crop, fdep_crop, mdep_crop)

        fndep_crop = min_crop_val
        mndep_crop = min_crop_val
        fdep_crop = min_crop_val
        mdep_crop = min_crop_val

    num_fndep = len(lengths['fndep'])
    num_fdep = len(lengths['fdep'])
    num_mndep = len(lengths['mndep'])
    num_mdep = len(lengths['mdep'])
    crops = {'fndep': [fndep_crop, num_fndep], 'fdep': [fdep_crop, num_fdep],
             'mndep': [mndep_crop, num_mndep], 'mdep': [mdep_crop, num_mdep]}

    indx_to_dict = {'fndep': 0, 'fdep': 1, 'mndep': 2, 'mdep': 3}

    return [crops, indx_to_dict]


def determine_crops(features, labels, config, gender, logger, mode_label):
    """
    Determines the shortest file for each subset of the data depending on the
    settings in config file. Linked to these values, the number of files is
    listed w.r.t. to the subsets: 'fndep', 'fdep', 'mndep', 'mdep'
    Inputs:
        features: Array of the data w.r.t the mode_label
        labels: Contains the labels (folder, class, score, gender, index) for
                the data w.r.t. mode_label
        config: config file holding state information for experiment
        gender: from config.USE_GENDER_WEIGHTS
        logger: records important information
        mode_label: set to 'train' or 'dev' or 'test' depending on data

    Outputs:
        min_samples: Dictionary, Key is 'fndep' / 'fdep' / 'mndep' / 'mdep',
                     Value is list containing the minimum sample length and
                     the number of files relating to the key
        indx_to_dict: Dictionary Key is same as min_samples, value is index 0-3
    """
    use_crop = config.EXPERIMENT_DETAILS['CROP']
    minimise_crop = config.MIN_CROP
    sub_sample = config.EXPERIMENT_DETAILS['SUB_SAMPLE_ND_CLASS']

    lengths, segments = get_lengths(features,
                                    labels,
                                    config)
    min_samples, indx_to_dict = crop_sections(lengths,
                                              config)
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
        dep_min = min(min_samples['fdep'][0], min_samples['mdep'][0])
        min_samples['fndep'][0] = ndep_min
        min_samples['mndep'][0] = ndep_min
        min_samples['fdep'][0] = dep_min
        min_samples['mdep'][0] = dep_min
    elif sub_sample and use_crop and gender and minimise_crop:
        pass
    elif minimise_crop and use_crop:
        min_val = get_min(min_samples)
        min_samples = set_min(min_samples,
                              min_val)

    return [min_samples, indx_to_dict]


def organise_data(config, logger, labels, database, mode_label='train',
                  hidden_test=False):
    """
    Loads the data and segments them according to the preferences in config
    file. This results in a feature array to be used for the experiments,
    along with corresponding updated folder labels, class and regressed
    scores, gender labels, and indices. Metadata are also collected such as the
    non-depressed / depressed indices, the length of every segmented file and
    the weights associated with the data partitions (i.e. non_dep vs dep)
    Inputs:
        config: config file holding state information for experiment
        logger: records important information
        labels: holds the folder, class, score, gender, index
        database: path to the database of raw data
        mode_label: set to 'train' or 'dev' or 'test' depending on data

    Outputs:
        features: Updated array of features
        labels: Updated lists containing the folders, classes, scores, gender,
                       and indices after segmentation
        index: list of the indices of all the ndep and dep files
        loc: List of the start-end length of every segmented data
        zeros: Dictionary, key is the folder, value are the indices related
               to that folder. Can be presented in a list if
               USE_GENDER_WEIGHTS=True in config file. Where zeros[0] is
               female and zeros[1] is male
        ones: Dictionary, key is the folder, value are the indices related
              to that folder. Can be presented in a list if
              USE_GENDER_WEIGHTS=True in config file. Where ones[0] is
              female and ones[1] is male
        weights: Dictionary, key is the folder, value is the respective weight
        set_weights: List of the weight for each subset of data.
    """
    freq_bins = config.EXPERIMENT_DETAILS['FREQ_BINS']
    feature_exp = config.EXPERIMENT_DETAILS['FEATURE_EXP']
    feature_dim = config.EXPERIMENT_DETAILS['FEATURE_DIMENSIONS']
    gender = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    features = util.load_data(database,
                              labels)
    if hidden_test:
        min_samples = {'fdep': [0, 0], 'mdep': [0, 0]}
    else:
        min_samples = determine_crops(features,
                                      labels,
                                      config,
                                      gender,
                                      logger,
                                      mode_label)

    features, labels, loc = process_data(freq_bins,
                                         features,
                                         labels,
                                         mode_label,
                                         min_samples,
                                         logger,
                                         feature_dim,
                                         feature_exp,
                                         config)

    if gender and config.EXPERIMENT_DETAILS['CROP']:
        # index in form [[fem_0, male_0], [fem_1, male_1]]
        zeros, zero_index, ones, one_index, weights, set_weights = \
            data_info_gender(labels,
                             mode_label,
                             logger,
                             config,
                             hidden_test)
    else:
        # index in form [0, 1]
        zeros, zero_index, ones, one_index, weights, set_weights = data_info(
            labels,
            mode_label,
            logger,
            config,
            hidden_test)

    index = [zero_index, one_index]

    return features, labels, index, loc, (zeros, ones, weights, set_weights)


def log_train_data(logger, folder_weights, partition_weights=None,
                   train_index=None):
    """
    Used to log the data from the training set for the case when
    config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']=True - list the
    corresponding weights for the partitions, 'fndep', 'fdep', 'mndep', 'mdep'

    Inputs:
        logger:
        folder_weights: Dictionary Key is folder, Value is respective weight
                        partition_weights: List of weights per 'fndep', 'fdep',
                        'mndep', 'mdep' or it is None
        train_index: List of indices for each partition of training data
    """
    logger.info(f"\nThe per class class weights (Non-Depresed vs Depressed) "
                f"are: {folder_weights}")
    if partition_weights is not None:
        female_total = len(train_index[0]) + len(train_index[1])
        male_total = len(train_index[2]) + len(train_index[3])
        nd_total = len(train_index[0]) + len(train_index[2])
        d_total = len(train_index[1]) + len(train_index[3])
        comp = female_total + male_total
        table = np.array([[len(train_index[0]), len(train_index[1]),
                           female_total], [len(train_index[2]),
                                           len(train_index[3]), male_total],
                          [nd_total, d_total, comp]])
        p = pd.DataFrame(data=table, index=['Female', 'Male', 'Total'],
                         columns=['Non-Dep', 'Dep', 'Total'])
        logger.info(f"\n{p}\n")
        logger.info(f"\nThe Gender Weights are: \nFemale_Non_Dep: "
                    f"{partition_weights[0]}\nFemale_Dep: "
                    f"{partition_weights[1]}\nMale_Non_Dep: "
                    f"{partition_weights[2]}\nMale_Dep: "
                    f"{partition_weights[3]}\n")


def run_train(config, logger, checkpoint, features_dir, data_saver, val=True):
    """
    High level function to process the training and validation data. This
    function obtains the file locations, folds for training/validation sets,
    processes the data to be used for training.
    Inputs:
        config: Config file holding state information for experiment
        logger: Logger to record important information
        checkpoint: Is there a checkpoint to load from?
        features_dir: Path to summary.pickle and database.h5 files
        data_saver: Contains the mean and std of the data if loading from a
                    checkpoint
        val: if True we are training with a validation set

    Outputs:
        generator: Generator for training and validation batch data loading
        class_data: Contains metadata including, list of dicts of indices for
        ndep / dep files for train and dev, contains dict of weights and a
        list of weights
    """
    train_file, dev_file, summary_file, database = file_paths(features_dir,
                                                              config)
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    logger.info(f"The dimensions of the logmel features before segmentation "
                f"are: {summary[1][-1]}")

    def data_to_array(file):
        data = util.csv_read(file)
        data_array = np.array([[d[0] for d in data], [d[1] for d in data],
                               [d[2] for d in data], [d[3] for d in
                                                      data]]).astype(int)
        return data_array, data

    train_labels, train_data = data_to_array(train_file)
    dev_labels, dev_data = data_to_array(dev_file)
    test_labels, test_data = data_to_array(config.TEST_SPLIT_PATH)

    comp_data = [j for i in [train_data, dev_data, test_data] for j in i]
    comp_data.sort()

    # Create empty indices arrays for train, dev, and test data. For each
    # array, fill with the indices relative to the complete dataset (189 files)
    train_indices = np.zeros((train_labels.shape[1])).astype(int)
    dev_indices = np.zeros((dev_labels.shape[1])).astype(int)
    test_indices = np.zeros((test_labels.shape[1])).astype(int)
    for p, i in enumerate(comp_data):
        h = np.where(train_labels[0] == int(i[0]))
        if len(h[0]) == 0:
            h = np.where(dev_labels[0] == int(i[0]))
            if len(h[0]) == 0:
                pass
            else:
                dev_indices[h[0][0]] = p
        else:
            train_indices[h[0][0]] = p

    # format: [folder, class, score, gender, index]
    train_labels = np.concatenate((train_labels, train_indices.reshape(1, -1)))
    dev_labels = np.concatenate((dev_labels, dev_indices.reshape(1, -1)))

    if not val:
        train_labels = np.hstack((train_labels, dev_labels))

    # class data is tuple (Dict(zeros), Dict(ones), Dict(weights), set_weights)
    train_features, train_labels, train_index, train_loc, class_data = \
        organise_data(config,
                      logger,
                      train_labels,
                      database,
                      mode_label='train')

    if val:
        dev_features, dev_labels, dev_index, dev_loc, _ = organise_data(config,
                                                                        logger,
                                                                        dev_labels,
                                                                        database,
                                                                        mode_label='dev')
    else:
        dev_features = dev_labels = dev_loc = 0
        dev_index = [[], []]

    gender_balance = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    if gender_balance:
        female_ndep_ind = train_index[0][0]
        male_ndep_ind = train_index[0][1]
        female_dep_ind = train_index[1][0]
        male_dep_ind = train_index[1][1]

        train_index = [female_ndep_ind, female_dep_ind, male_ndep_ind,
                       male_dep_ind]
        log_train_data(logger,
                       class_data[-2],
                       class_data[-1],
                       train_index)

        if val:
            dev_index = [dev_index[0][0], dev_index[1][0], dev_index[0][1],
                         dev_index[1][1]]
            # For use later, set the labels to 0 (fnd), 1 (fd), 2 (mnd), and 3 (md)
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
            dev_index = [[], [], [], []]
    else:
        log_train_data(logger,
                       class_data[-1])

    zeros, ones, weights, set_weights = class_data
    if val:
        dev_weights = {}
        for i in range(len(dev_labels[0])):
            folder = dev_labels[0][i]
            clss = dev_labels[1][i]
            gender = dev_labels[3][i]
            if clss == 0 or clss == 2:
                if gender_balance and gender == 0:
                    dev_weights[folder] = set_weights[0]
                elif gender_balance and gender == 1:
                    dev_weights[folder] = set_weights[2]
                else:
                    dev_weights[folder] = set_weights[0]
            else:
                if gender_balance and gender == 0:
                    dev_weights[folder] = set_weights[1]
                elif gender_balance and gender == 1:
                    dev_weights[folder] = set_weights[3]
                else:
                    dev_weights[folder] = set_weights[1]
        class_data = zeros, ones, weights, set_weights, dev_weights
    else:
        class_data = zeros, ones, weights, set_weights, {}

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


def run_test(config, logger, checkpoint, features_dir, data_saver,
             tester=False, hidden_test=False):
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
        hidden_test: Bool, set True if we know the test labels

    Outputs:
        generator: Generator for training and validation batch data loading
        class_weights_train:
        class_weights_dev
        zero_train
        one_train
    """
    train_file, dev_file, summary_file, database = file_paths(features_dir,
                                                              config)
    test_file = config.TEST_SPLIT_PATH
    with open(summary_file, 'rb') as f:
        summary = pickle.load(f)
    logger.info(f"The dimensions of the logmel features before segmentation "
                f"are: {summary[1][-1]}")

    def data_to_array(file):
        data = util.csv_read(file)
        data_array = np.array([[d[0] for d in data], [d[1] for d in data],
                               [d[2] for d in data], [d[3] for d in
                                                      data]]).astype(int)
        return data_array, data

    train_labels, train_data = data_to_array(train_file)
    dev_labels, dev_data = data_to_array(dev_file)
    test_labels, test_data = data_to_array(config.TEST_SPLIT_PATH)

    comp_data = [j for i in [train_data, dev_data, test_data] for j in i]
    comp_data.sort()

    train_indices = np.zeros((train_labels.shape[1])).astype(int)
    dev_indices = np.zeros((dev_labels.shape[1])).astype(int)
    test_indices = np.zeros((test_labels.shape[1])).astype(int)
    for p, i in enumerate(comp_data):
        h = np.where(train_labels[0] == int(i[0]))
        if len(h[0]) == 0:
            h = np.where(dev_labels[0] == int(i[0]))
            if len(h[0]) == 0:
                h = np.where(test_labels[0] == int(i[0]))
                test_indices[h[0][0]] = p
            else:
                dev_indices[h[0][0]] = p
        else:
            train_indices[h[0][0]] = p

    train_labels = np.concatenate((train_labels, train_indices.reshape(1, -1)))
    dev_labels = np.concatenate((dev_labels, dev_indices.reshape(1, -1)))
    test_labels = np.concatenate((test_labels, test_indices.reshape(1, -1)))

    if tester:
        mode_lab = 'test'
        labs = test_labels
    else:
        mode_lab = 'dev'
        labs = dev_labels
    # if gender_balance index = [[fem_0, male_0], [fem_1, male_1]]
    # else index = [0, 1]
    features, labels, index, loc, class_data = organise_data(config,
                                                             logger,
                                                             labs,
                                                             database,
                                                             mode_label=mode_lab,
                                                             hidden_test=hidden_test)

    gender_balance = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']

    zeros, ones, weights, set_weights = class_data
    if gender_balance:
        f_ndep_ind = index[0][0]
        m_ndep_ind = index[0][1]
        f_dep_ind = index[1][0]
        m_dep_ind = index[1][1]

        index = [f_ndep_ind, f_dep_ind, m_ndep_ind, m_dep_ind]
        for p, i in enumerate(labels[3]):
            if labels[3][p] == 0 and i == 0:
                pass
            elif labels[3][p] == 0 and i == 1:
                labels[3][p] = 2
            elif labels[3][p] == 1 and i == 0:
                labels[3][p] = 1
            else:
                labels[3][p] = 3
    split_by_gender = config.EXPERIMENT_DETAILS['SPLIT_BY_GENDER']
    if split_by_gender:
        if not gender_balance:
            f_ndep_ind, f_dep_ind, m_ndep_ind, m_dep_ind = per_gender_indices(labels)
        female = [f_ndep_ind, f_dep_ind]
        male = [m_ndep_ind, m_dep_ind]

        gen_female = data_gen.GenerateData(train_labels=None,
                                           dev_labels=labels,
                                           train_feat=None,
                                           dev_feat=features,
                                           train_loc=None,
                                           dev_loc=loc,
                                           train_indices=index,
                                           dev_indices=female,
                                           logger=logger,
                                           config=config,
                                           checkpoint=checkpoint,
                                           gender_balance=False,
                                           data_saver=data_saver)
        gen_male = data_gen.GenerateData(train_labels=None,
                                         dev_labels=labels,
                                         train_feat=None,
                                         dev_feat=features,
                                         train_loc=None,
                                         dev_loc=loc,
                                         train_indices=index,
                                         dev_indices=male,
                                         logger=logger,
                                         config=config,
                                         checkpoint=checkpoint,
                                         gender_balance=False,
                                         data_saver=data_saver)
        return (gen_female, gen_male), (0, 0, weights, set_weights, 0)
    else:
        generator = data_gen.GenerateData(train_labels=None,
                                          dev_labels=labels,
                                          train_feat=None,
                                          dev_feat=features,
                                          train_loc=None,
                                          dev_loc=loc,
                                          train_indices=index,
                                          dev_indices=index,
                                          logger=logger,
                                          config=config,
                                          checkpoint=checkpoint,
                                          gender_balance=gender_balance,
                                          data_saver=data_saver)

    return generator, (0, 0, weights, set_weights, 0)

