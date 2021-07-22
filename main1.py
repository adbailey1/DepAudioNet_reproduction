import logging
import logging.handlers
import argparse
import os
import time
from data_loader import organiser
import shutil
import sys
import torch
import random
import numpy as np
import pandas as pd
import utilities.utilities_main as util
import socket
from distutils.dir_util import copy_tree
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import natsort
from utilities import model_utilities as mu
from exp_run.plotter import plot_graph
from exp_run import config_dataset
import pickle
from exp_run.models_pytorch import CustomMel7 as CustomMel
from exp_run.models_pytorch import CustomRaw1 as CustomRaw
from exp_run import config_1 as config
learn_rate_factor = 2
EPS = 1e-12


def evaluation_for_test(results_dict, num_class, f_score, main_logger):
    """
    This function is only used for mode==test and data_type=='test'. Every
    prediction for each folder in the test set is accumulated to results_dict
    along with the number of instances of each folder. There will be multiple
    predictions for each folder depending on how many times the experiment
    was run during training (e.g. 5). If the user sets argument:
    prediction_metric=0 -> the accuracy will be determined for every experiment
    iteration (e.g. 5) and the best performing model will be selected.
    prediction_metric=1 -> the average of the accumulated predictions for
    each folder will be taken and the final score will relate to these
    averaged results.
    prediction_metric=2 -> The majority vote of the accumulated predictions
    for each folder will be taken and the final score will related to these
    results
    Input
        results_dict: dictionary key: output, target, accum. For each of
                      these keys is a corresponding dictionary where key:
                      folder, value relates to the super key: output ->
                      predictions from experiments, target -> corresponding
                      label for the folder, accum -> the accumulated
                      instances of each folder
        num_class: int - Number of classes in the dataset
        f_score: str - Type of F1 Score processing

    Outputs:
        scores: List - contains accuracy, fscore and tn_fp_fn_tp
    """
    temp_tar = np.array(list(results_dict['target'].values()))
    final_results = {}
    # Pick best performing model
    if prediction_metric == 0:
        f_score_avg = []
        temp_out = np.zeros((exp_runthrough,
                             len(results_dict['output'].keys())))
        temp_scores = np.zeros((exp_runthrough, 15))
        for pos, f in enumerate(results_dict['output'].keys()):
            temp_out[:, pos] = list(results_dict['output'][f] /
                                    results_dict['accum'][f])
        for exp in range(exp_runthrough):
            temp_scores[exp, :], _ = prediction_and_accuracy(temp_out[exp, :],
                                                             temp_tar,
                                                             True,
                                                             num_class,
                                                             np.zeros(15),
                                                             0,
                                                             0,
                                                             f_score)
            f_score_avg = f_score_avg + [np.mean(temp_scores[exp, 6:8])]
        best_result_index = f_score_avg.index(max(f_score_avg))
        print(f"\nThe best performing model was experiment: "
              f"{best_result_index+1}")
        main_logger.info(f"The best performing model was experiment: "
                         f"{best_result_index + 1}")
        scores = temp_scores[best_result_index, :]
    # Average the performance of all models
    elif prediction_metric == 1:
        print("\nPerforming averaging of accumulated predictions")
        for f in results_dict['output'].keys():
            final_results[f] = np.average(
                results_dict['output'][f] / results_dict['accum'][f])
        temp_out = np.array(list(final_results.values()))
        scores, _ = prediction_and_accuracy(temp_out,
                                            temp_tar,
                                            True,
                                            num_class,
                                            np.zeros(15),
                                            0,
                                            0,
                                            f_score)
    # Calculate majority vote from all models
    elif prediction_metric == 2:
        print("\nPerforming majority vote on accumulated predictions")
        for f in results_dict['output'].keys():
            final_results[f] = np.average(
                np.round(results_dict['output'][f] / results_dict['accum'][f]))
        temp_out = np.array(list(final_results.values()))
        scores, _ = prediction_and_accuracy(temp_out,
                                            temp_tar,
                                            True,
                                            num_class,
                                            np.zeros(15),
                                            0,
                                            0,
                                            f_score)
    scores[8] = np.mean(scores[0:2])
    scores[9] = np.mean(scores[6:8])
    scores = [scores[8], scores[0], scores[1], scores[9], scores[6], scores[7],
              scores[11], scores[12], scores[13], scores[14]]

    return scores


def calculate_accuracy(target, predict, classes_num, f_score_average):
    """
    Calculates accuracy, precision, recall, F1-Score, True Negative,
    False Negative, True Positive, and False Positives of the output of
    the model

    Inputs
      target: np.array() The labels for the predicted outputs from the model
      predict: np.array() The batched outputs of the network
      classes_num: int How many classes are in the dataset
      f_score_average: str How to average the F1-Score

    Outputs:
      accuracy: Float Accuracy of the model outputs
      p_r_f: Array of Floats Precision, Recall, and F1-Score
      tn_fp_fn_tp: Array of Floats True Negative, False Positive,
                   False Negative, and True Positive
    """

    number_samples_labels = len(target)

    number_correct_predictions = np.zeros(classes_num)
    total = np.zeros(classes_num)

    for n in range(number_samples_labels):
        total[target[n]] += 1

        if target[n] == predict[n]:
            number_correct_predictions[target[n]] += 1

    con_matrix = confusion_matrix(target,
                                  predict)
    tn_fp_fn_tp = con_matrix.ravel()
    if tn_fp_fn_tp.shape != (4,):
        value = int(tn_fp_fn_tp)
        if target[0][0] == 1:
            tn_fp_fn_tp = np.array([0, 0, 0, value])
        elif target[0][0] == 0:
            tn_fp_fn_tp = np.array([value, 0, 0, 0])
        else:
            print('Error in the true_neg/false_pos value')
            sys.exit()

    if f_score_average is None:
        # This code fixes the divide by zero error
        accuracy = np.divide(number_correct_predictions,
                             total,
                             out=np.zeros_like(number_correct_predictions),
                             where=total != 0)
        p_r_f = metrics.precision_recall_fscore_support(target,
                                                        predict)
    elif f_score_average == 'macro':
        # This code fixes the divide by zero error
        accuracy = np.divide(number_correct_predictions,
                             total,
                             out=np.zeros_like(number_correct_predictions),
                             where=total != 0)
        p_r_f = metrics.precision_recall_fscore_support(target,
                                                        predict,
                                                        average='macro')
    elif f_score_average == 'micro':
        # This code fixes the divide by zero error
        accuracy = np.divide(np.sum(number_correct_predictions),
                             np.sum(total),
                             out=np.zeros_like(number_correct_predictions),
                             where=total != 0)
        p_r_f = metrics.precision_recall_fscore_support(target,
                                                        predict,
                                                        average='micro')
    else:
        raise Exception('Incorrect average!')

    if p_r_f[0].shape == (1,):
        temp = np.zeros((4, 2))
        position = int(target[0])
        for val in range(len(p_r_f)):
            temp[val][position] = float(p_r_f[val])

        p_r_f = (temp[0], temp[1], temp[2], temp[3])

    return accuracy, p_r_f, tn_fp_fn_tp


def forward(model, generate_dev, data_type):
    """
    Pushes the data to the model and collates the outputs

    Inputs:
        model: The neural network for experimentation
        generate_dev: generator - holds the batches for the validation
        data_type: str - set to 'train', 'dev' or 'test'

    Output:
        results_dict: dictionary - Outputs, optional - labels and folders
    """
    outputs = []
    folders = []
    targets = []

    # Evaluate on mini-batch
    counter = 0
    for data in generate_dev:
        (batch_data, batch_label, batch_folder, batch_locator) = data

        # Predict
        model.eval()

        # Potentially speeds up evaluation and memory usage
        with torch.no_grad():
            batch_output = get_output_from_model(model=model,
                                                 data=batch_data)
        counter += 1
        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        folders.append(batch_folder)
        targets.append(batch_label)

    results_dict = {}

    outputs = np.concatenate(outputs, axis=0)
    results_dict['output'] = outputs
    folders = np.concatenate(folders, axis=0)
    results_dict['folder'] = np.array(folders)
    targets = np.concatenate(targets, axis=0)
    results_dict['target'] = np.array(targets)

    return results_dict


def evaluate(model, generator, data_type, class_weights, comp_res, class_num,
             f_score_average, epochs, logger, gender_balance=False):
    """
    Processes the validation set by creating batches, passing these through
    the model and then calculating the resulting loss and accuracy metrics.

    Input
        model: neural network for experimentation
        generator: generator - Created to load validation data to the model
        data_type: str - set to 'dev' or 'test'
        class_weights: Dictionary - Key is Folder, Value is weight
        comp_res: np.array - holds results for each iteration of experiment
        class_num: int - Number of classes in the dataset
        f_score_average: str - Type of F1 Score processing
        averaging: str - Geometric or arithmetic for the outputs from the model
        net_p: dictionary - holds the model configurations
        recurrent_out: str - If RNN used, how to process the output
        epochs: int - The current epoch
        logger: log for keeping important info
        gender_balance: bool

    Returns:
        complete_results: np.array - holds results for each iteration of
                                     experiment
        per_epoch_pred: numpy.array - collated outputs and labels from the
                        current validation test
    """
    # Generate function
    print('Generating data for evaluation')
    start_time_dev = time.time()
    if data_type == 'dev':
        generate_dev = generator.generate_development_data(epoch=epochs)
    else:
        generate_dev = generator.generate_test_data()

    # Forward
    results_dict = forward(model=model,
                           generate_dev=generate_dev,
                           data_type=data_type)

    outputs = results_dict['output']  # (audios_num, classes_num)
    folders = results_dict['folder']
    targets = results_dict['target']  # (audios_num, classes_num)

    collected_output = {}
    output_for_loss = {}
    collected_targets = {}
    counter = {}
    for p, fol in enumerate(folders):
        if fol not in collected_output.keys():
            collected_targets[fol] = targets[p]
            output_for_loss[fol] = outputs[p]
            collected_output[fol] = np.round(outputs[p])
            counter[fol] = 1
        else:
            output_for_loss[fol] = outputs[p]
            collected_output[fol] += np.round(outputs[p])
            counter[fol] += 1

    if mode == 'test':
        intermediate_outputs = collected_output
        intermediate_counter = counter
        intermediate_targets = collected_targets
    new_outputs = []
    new_folders = []
    new_targets = []
    new_output_for_loss = []
    # This is essentially performing majority vote. We have rounded all the
    # predictions made per file. We now divide the resulting value by the
    # number of instances per file.
    for co in collected_output:
        tmp = collected_output[co] / counter[co]
        new_outputs.append(tmp)
        tmp = output_for_loss[co] / counter[co]
        new_output_for_loss.append(tmp)
        new_folders.append(co)
        new_targets.append(collected_targets[co])

    outputs = np.array(new_outputs)
    folders = np.array(new_folders)
    targets = np.array(new_targets)

    calculate_time(start_time_dev,
                   time.time(),
                   data_type,
                   logger)

    batch_weights = find_batch_weights(folders,
                                       class_weights)
    loss = mu.calculate_loss(torch.Tensor(outputs),
                             torch.LongTensor(targets),
                             batch_weights,
                             gender_balance)

    if gender_balance:
        targets = targets % 2
    complete_results, per_epoch_pred = prediction_and_accuracy(outputs,
                                                               targets,
                                                               True,
                                                               class_num,
                                                               comp_res,
                                                               loss, 0,
                                                               f_score_average)
    if data_type == 'test':
        return complete_results, (per_epoch_pred, intermediate_outputs, \
               intermediate_counter, intermediate_targets)
    else:
        return complete_results, per_epoch_pred


def logging_info(current_dir, data_type=''):
    """
    Sets up the logger to be used for the current experiment. This is useful
    to capture relevant information during the course of the experiment.

    Inputs:
        current_dir: str - the location of the current experiment
        data_type: str - set to 'test' or 'dev' when running the code in test
                         mode. 'dev' will load existing best epochs and
                         re-run them on validation set, 'test' will load
                         existing best epochs and run them on test set.

    Output
        main_logger: logger - The created logger
    """
    if mode == 'test':
        if data_type == 'test':
            log_path = os.path.join(current_dir, "test.log")
        elif data_type == 'dev':
            log_path = os.path.join(current_dir, 'log',
                                    f"model_test.log")
    else:
        log_path = os.path.join(current_dir, 'log',
                                f"model_{folder_extensions[i]}.log")
    main_logger = logging.getLogger('MainLogger')
    main_logger.setLevel(logging.INFO)
    if os.path.exists(log_path) and mode == 'test':
        os.remove(log_path)
    main_handler = logging.handlers.RotatingFileHandler(log_path)
    main_logger.addHandler(main_handler)

    main_logger.info(config_dataset.SEPARATOR)
    main_logger.info('EXPERIMENT DETAILS')
    for dict_val in config.EXPERIMENT_DETAILS:
        if dict_val == 'SEED':
            main_logger.info(f"Starting {dict_val}:"
                             f" {str(config.EXPERIMENT_DETAILS[dict_val])}")
        else:
            main_logger.info(f"{dict_val}:"
                             f" {str(config.EXPERIMENT_DETAILS[dict_val])}")
    main_logger.info(f"Current Seed: {chosen_seed}")
    main_logger.info(f"Logged into: {socket.gethostname()}")
    main_logger.info(f"Experiment details: {config.EXPERIMENT_BRIEF}")
    main_logger.info(config_dataset.SEPARATOR)

    return main_logger


def create_model():
    """
    Creates the model to be used in the current experimentation

    Output
        model: obj - The model to be used for training during experiment
    """
    if config.EXPERIMENT_DETAILS['FEATURE_EXP'] == 'mel':
        model = CustomMel()
    else:
        model = CustomRaw()
    if cuda:
        model.cuda()

    return model


def setup(current_dir, model_dir, data_type='', path_to_logger_for_test=None):
    """
    Creates the necessary directories, data folds, logger, and model to be
    used in the experiment. It also determines whether a previous checkpoint
    has been saved.

    Inputs:
        current_dir: str - dir for the experiment
        model_dir: str - location of the current model run-through
        data_type: str - set to 'test' for different setup processing
        path_to_logger_for_test: str - path to create a logger for running a
                                       test

    Outputs
        main_logger: logger - The logger to be used to record information
        model: obj - The model to be used for training during the experiment
        checkpoint_run: str - The location of the last saved checkpoint
        checkpoint: bool - True if loading from a saved checkpoint
        next_exp: bool - If loading from a checkpoint is suspected but the
                         current experiment has been completed set True
    """
    reproducibility(chosen_seed)
    checkpoint_run = None
    checkpoint = False
    next_exp = False
    if not os.path.exists(features_dir):
        print('There is no folder and therefore no database created. '
              'Create the database first')
        sys.exit()
    if os.path.exists(current_dir) and os.path.exists(model_dir) and debug:
        shutil.rmtree(current_dir, ignore_errors=False, onerror=None)
    # THIS WILL DELETE EVERYTHING IN THE CURRENT WORKSPACE #
    if os.path.exists(current_dir) and os.path.exists(model_dir):
        temp_dirs = os.listdir(model_dir)
        temp_dirs = natsort.natsorted(temp_dirs, reverse=True)
        temp_dirs = [d for d in temp_dirs if '.pth' in d]
        if len(temp_dirs) == 0:
            pass
        else:
            if int(temp_dirs[0].split('_')[1]) == final_iteration and mode ==\
                    'train':
                directory = model_dir.split('/')[-1]
                final_directory = model_dir.replace(directory, str(exp_runthrough))
                if os.path.exists(final_directory):
                    temp_dirs2 = os.listdir(final_directory)
                    temp_dirs2 = natsort.natsorted(temp_dirs2, reverse=True)
                    temp_dirs2 = [d for d in temp_dirs2 if '.pth' in d]
                    if int(temp_dirs2[0].split('_')[1]) == final_iteration:
                        if i == exp_runthrough-1:
                            print(f"A directory at this location "
                                  f"exists: {current_dir}")
                            sys.exit()
                        else:
                            next_exp = True
                            return None, None, None, None, next_exp
                    else:
                        return None, None, None, None, next_exp
                else:
                    return None, None, None, None, next_exp
            else:
                print(f"Current directory exists but experiment not finished")
                print(f"Loading from checkpoint: "
                      f"{int(temp_dirs[0].split('_')[1])}")
                checkpoint_run = os.path.join(model_dir, temp_dirs[0])
                checkpoint = True
    elif not os.path.exists(current_dir):
        os.mkdir(current_dir)
        util.create_directories(current_dir,
                                config.EXP_FOLDERS)
        os.mkdir(model_dir)
    elif os.path.exists(current_dir) and not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if mode == 'test' and path_to_logger_for_test is not None and data_type \
            == 'test':
        if os.path.exists(path_to_logger_for_test):
            shutil.rmtree(path_to_logger_for_test, ignore_errors=False,
                          onerror=None)
        os.mkdir(path_to_logger_for_test)
        main_logger = logging_info(path_to_logger_for_test,
                                   data_type)
    else:
        main_logger = logging_info(current_dir,
                                   data_type)

    model = create_model()

    return main_logger, model, checkpoint_run, checkpoint, next_exp


def record_top_results(current_results, scores, epoch):
    """
    Function to record the best validation F1-Score up to the current epoch.
    More accurate than the alternate function record_top_results

    Inputs:
        current_results: list - current epoch results
        scores: tuple - contains the best results for the experiment
        epoch: int - The current epoch

    Output
        best_res: list - updated best result and epoch of discovery
    """
    if current_results[8] > .86:
        if validate:
            train_f = current_results[9] / 4
            train_loss = current_results[10] / 10
            dev_f = current_results[-6]
            dev_loss = current_results[-5] / 10
            total = train_f - train_loss + dev_f - dev_loss
            if total > scores[0]:
                best_res = [total, current_results[8], current_results[0],
                            current_results[1], current_results[9],
                            current_results[6], current_results[7],
                            current_results[10], current_results[23],
                            current_results[15], current_results[16], dev_f,
                            current_results[21], current_results[22],
                            current_results[25], epoch]
            else:
                best_res = scores
        else:
            if current_results[8] > scores[0]:
                best_res = [current_results[8], current_results[8],
                            current_results[0],
                            current_results[1], current_results[9],
                            current_results[6], current_results[7],
                            current_results[10], 0, 0, 0, 0, 0, 0, 0, epoch]
            else:
                best_res = scores
    else:
        best_res = scores

    return best_res


def initialiser(test_value):
    """
    Used to set a bool to True for the initialisation of some function or
    variable

    Input
        test_value: int - If set to 1 then this is the initial condition
                    otherwise, already initialised

    Output
        bool - True if this is the initialisation case
    """
    if test_value == 1:
        return True
    else:
        return False


def compile_train_val_pred(train_res, val_res, comp_train, comp_val, epoch):
    """
    Used to group the latest results for both the training and the validation
    set into their respective complete results array

    Inputs
         train_res: numpy.array - The current results for this epoch
         val_res: numpy.array - The current results for this epoch
         comp_train: numpy.array - The total recorded results
         comp_val: numpy.array - The total recorded results
         epoch: int - The current epoch used for initialisation

    Outputs
        comp_train: numpy.array - The updated complete results
        comp_val - numpy.array - The updated complete results
    """
    # 3D matrix ('Num_segments_batches', 'pred+label', 'epochs')
    if epoch == 1:
        comp_train = train_res
        comp_val = val_res
    else:
        if train_res.shape[0] != comp_train.shape[0]:
            difference = comp_train.shape[0] - train_res.shape[0]

            train_res = np.vstack((train_res, np.zeros((difference, 2))))

        comp_train = np.dstack((comp_train, train_res))
        comp_val = np.dstack((comp_val, val_res))

    return comp_train, comp_val


def update_complete_results(complete_results, avg_counter, placeholder,
                            best_scores):
    """
    Finalises the complete results dataframe by calculating the mean of the 2
    class scores for accuracy and F1-Score and in the case of the training
    data, divides the results by the number of iterations in order to get the
    average results from the current epoch (previously updated by accumulation)
    Also obtains the best scores for the model.

    Inputs
        complete_results: dataframe - holds the complete results from the
                          experiment so far
        avg_counter: int - used in train mode to average the recorded results
                     for the current epoch
        placeholder: Essentially the number of epochs (but can be used in
                     iteration mode)
        best_scores: list - Contains the best scores so far and the
                     respective epochs, calculated according to weighting of
                     training and validation scores/losses:

                     train_f/4 - train_loss/10 + dev_f - dev_loss/10

                     with a threshold of 86% if training score is less than
                     this we do not consider it

    Outputs
        complete_results: np.array - Updated version of the complete results
        best_scores: list - Updated version of best_scores
    """
    complete_results[0:11] = complete_results[0:11] / avg_counter
    # Accuracy Mean
    complete_results[8] = np.mean(complete_results[0:2])
    complete_results[23] = np.mean(complete_results[15:17])
    # FScore Mean
    complete_results[9] = np.mean(complete_results[6:8])
    complete_results[24] = np.mean(complete_results[21:23])
    print_log_results(placeholder, complete_results[0:15], 'train')
    if validate:
        print_log_results(placeholder, complete_results[15:], 'dev')

    best_scores = record_top_results(complete_results,
                                     best_scores,
                                     placeholder)

    return complete_results, best_scores


def prediction_and_accuracy(batch_output, batch_labels, initial_condition,
                            num_of_classes, complete_results, loss,
                            per_epoch_pred, f_score_average=None):
    """
    Calculates the accuracy (including F1-Score) of the predictions from a
    model. Also the True Negatives, False Negatives, True Positives, and False
    Positives are calculated. These results are stored along with results
    from previous epochs.

    Input
        batch_output: The output from the model
        batch_labels: The respective labels for the batched output
        initial_condition: Bool - True if this is the first instance to set
                           up the variables for logging accuracy
        num_of_classes: The number of classes in this dataset
        complete_results: np.array - holds results for each iteration of
                                     experiment
        loss: The value of the loss from the current epoch
        per_epoch_pred: Combined batch outputs and labels for record keeping
        f_score_average: The type of averaging to be used fro the F1-Score (
                         Macro, Micro, or None

    Output
        complete_results: np.array - holds results for each iteration of
                                     experiment
        per_epoch_pred: Combined results of batch outputs and labels for
                        current epoch
    """
    if type(batch_output) is not np.ndarray:
        batch_output = batch_output.data.cpu().numpy()
        batch_labels = batch_labels.data.cpu().numpy()

    if len(batch_output.shape) == 1:
        batch_output = batch_output.reshape(-1, 1)
    if len(batch_labels.shape) == 1:
        batch_labels = batch_labels.reshape(-1, 1)
    if initial_condition:
        per_epoch_pred = np.hstack((batch_output, batch_labels))
    else:
        temp_stack = np.hstack((batch_output, batch_labels))
        per_epoch_pred = np.vstack((per_epoch_pred, temp_stack))

    prediction = np.round(batch_output)
    prediction = prediction.reshape(-1)

    if len(batch_labels.shape) > 1:
        batch_labels = batch_labels.reshape(-1)
    if batch_labels.dtype == 'float32':
        batch_labels = batch_labels.astype(np.long)

    acc, fscore, tn_fp_fn_tp = calculate_accuracy(batch_labels,
                                                  prediction,
                                                  num_of_classes,
                                                  f_score_average)
    complete_results[0:2] += acc
    complete_results[2:8] += np.array(fscore[0:3]).reshape(1, -1)[0]
    complete_results[10] += loss
    complete_results[11:15] += tn_fp_fn_tp

    return complete_results, per_epoch_pred


def print_log_results(epoch, results, data_type):
    """
    Used to print/log results after every epoch

    Inputs
        epoch: int - The current epoch
        results: numpy.array - The current results
        data_type: str - Set to train, val, or test
    """
    print('\n', config_dataset.SEPARATOR)
    print(f"{data_type} accuracy at epoch: {epoch}\n{data_type} Accuracy: Mean:"
          f" {np.round(results[8], 3)} - {np.round(results[0:2], 3)}, "
          f"F1_Score: Mean: {np.round(results[9], 3)} -"
          f" {np.round(results[6:8], 3)}, Loss: {np.round(results[10], 3)}")

    print(config_dataset.SEPARATOR, '\n')

    main_logger.info(f"\n{config_dataset.SEPARATOR}{config_dataset.SEPARATOR}")
    main_logger.info(f"{data_type} accuracy at epoch: {epoch}\n{data_type} "
                     f"Accuracy: Mean: {np.round(results[8], 3)} -"
                     f" {np.round(results[0:2], 3)}, F1_Score: Mean:"
                     f" {np.round(results[9], 3)},"
                     f" {np.round(results[6:8], 3)}, Loss:"
                     f" {np.round(results[10], 3)}")
    main_logger.info(f"{config_dataset.SEPARATOR}{config_dataset.SEPARATOR}\n")


def final_organisation(scores, train_pred, val_pred, df, patience, epoch,
                       workspace_files_dir, data_saver):
    """
    Records final information with the logger such as the best scores for
    training and validation and saves/copies files from the current
    experiment into the saved model directory for future analysis. The
    complete results to the current epoch are saved for checkpoints or future
    analysis.

    Copys the current directory to the current experiment dir but only copies
    over the current main.py and config.py files in case multiple are present
    due to multiple experiments being run

    Inputs
        scores: list - The best scores from the training and validation results
        train_pred: numpy.array - Record of the complete outputs of the
                    network for every epoch
        val_pred: numpy.array - Record of the complete outputs of the
                  network for every epoch
        df: pandas.dataframe - The complete results for every epoch
        patience: int - Used to record if early stopping was implemented
        epoch: int - The current epoch
        workspace_files_dir: str - Location of the programme code
    """
    main_logger.info(f"Best epoch at: {scores[-1]}")
    main_logger.info(f"Best Train Acc: {scores[1]}\nBest Train Fscore:"
                     f" {scores[4]}\nBest Train Loss: {scores[7]}")
    if validate:
        main_logger.info(f"Best Val "
                     f"Acc: {scores[8]}\nBest Val Fscore: {scores[11]}\nBest "
                     f"Val Loss: {scores[14]}")

    main_logger.info(f"\nscores: {scores[1:-1]}")

    if epoch == final_iteration:
        main_logger.info(f"System will exit as the total number of "
                         f"epochs has been reached  {final_iteration}")
    else:
        main_logger.info(f"System will exit as the validation loss "
                         f"has not improved for {patience} epochs")
    print(f"System will exit as the validation loss has not "
          "improved for {patience} epochs")
    util.save_model_outputs(model_dir,
                            df,
                            train_pred,
                            val_pred,
                            scores,
                            data_saver)

    copy_tree(workspace_files_dir, current_dir+'/daic')
    dirs = os.listdir(workspace_files_dir)
    current_main = 'main' + str(position) + '.py'
    mains = [d for d in dirs if 'main' in d and d != current_main]

    current_config = 'config_' + str(position) + '.py'
    nest = 'exp_run'
    dirs = os.listdir(os.path.join(workspace_files_dir, nest))
    configs = [d for d in dirs if 'config_' in d and d != current_config]
    del configs[configs.index('config_dataset.py')]

    for m in mains:
        os.remove(os.path.join(current_dir, 'daic', m))
    for c in configs:
        os.remove(os.path.join(current_dir, 'daic', nest, c))


def reduce_learning_rate(optimizer):
    """
    Reduce the learning rate of the optimiser for training

    Input
        optimiser: obj - The optimiser setup at the start of the experiment
    """
    learning_rate_reducer = 0.9
    for param_group in optimizer.param_groups:
        print('Reducing Learning rate from: ', param_group['lr'],
              ' to ', param_group['lr'] * learning_rate_reducer)
        main_logger.info(f"Reducing Learning rate from: "
                         f"{param_group['lr']}, to "
                         f"{param_group['lr'] * learning_rate_reducer}")
        param_group['lr'] *= learning_rate_reducer


def reproducibility(chosen_seed):
    """
    The is required for reproducible experimentation. It sets the random
    generators for the different libraries used to a specific, user chosen
    seed.

    Input
        chosen_seed: int - The seed chosen for this experiment
    """
    torch.manual_seed(chosen_seed)
    torch.cuda.manual_seed_all(chosen_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(chosen_seed)
    random.seed(chosen_seed)


def get_output_from_model(model, data):
    """
    Pushes the batched data to the user specified neural network. The output
    is pushed back to the CPU if GPU is being used for training.

    Inputs:
        model: obj - the neural network for experimentation
        data: Data to be pushed to the model

    Output
        output: The output of the model from the input batch data
    """
    current_data = mu.create_tensor_data(data,
                                         cuda)

    output = model(current_data)
    if cuda:
        output = output.cpu()

    return output


def begin_evaluation(mode, epoch, reset, iteration, iteration_epoch):
    """
    Determines whether to start the validation processing. This is determined
    by the a change in epoch and reset

    Inputs
        mode: str - Is the mode in epoch selection or iteration selection
        epoch: int - The current epoch of the experiment
        reset: Bool - True if the end of a training phase has occurred
        iteration: int - The current iteration of the experiment
        iteration_epoch: int - The number of iterations equivalent to the
                         number of epochs if it were in epoch mode

    Output
        Bool: True if validation set should be processed
    """
    if mode == 'epoch':
        if epoch > 0 and reset:
            return True
    elif mode == 'iteration':
        it = iteration + 1
        if it % iteration_epoch == 0 and iteration > 0:
            return True
    elif mode is None:
        print('Wrong Mode Selected. Choose either epoch or iteration in the '
              'config file.')
        print('The program will now exit')
        sys.exit()


def calculate_time(start_time, end_time, mode_label, main_logger, placeholder=0,
                   iteration=0):
    """
    Stores the time it took for the current experimental iteration and saves
    it to the log

    Inputs:
        start_time: value of timer when started
        end_time: value of timer when stopped
        mode_label: 'train', 'dev', or 'test'
        main_logger: log to retain useful information
        placeholder: used in place of the current epoch
        iteration: the current training iteration
    """
    calc_time = end_time - start_time
    if mode_label == 'train':
        print(f"Iteration: {iteration}\nTime taken for {mode_label}:"
              f" {calc_time:.2f}s")
        main_logger.info(f"Time taken for {mode_label}: {calc_time:.2f}s "
                         f"at iteration: {iteration}, epoch: {placeholder}")
    else:
        print(f"\nTime taken to evaluate {mode_label}: {calc_time:.2f}s")
        main_logger.info(f"Time taken to evaluate {mode_label}:"
                         f" {calc_time:.2f}s")


def find_batch_weights(folders, weights):
    """
    Finds the corresponding weight for the folders in the current batch,
    if weights are not used as specified by config file, set the weights to '1'

    Inputs:
        folders: The folders in the current batch
        weights: Dictionary, Key is Folder, Value is corresponding weight
    Outputs:
        batch_weights: Weights w.r.t. folder for the current batch
    """
    batch_weights = torch.ones(folders.shape[0])
    use_weights = config.EXPERIMENT_DETAILS['CLASS_WEIGHTS'] or \
                  config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    for indx, folder in enumerate(folders):
        if use_weights:
            batch_weights[indx] = weights[folder]
        else:
            batch_weights[indx] = 1

    return batch_weights.reshape(-1, 1)


def bookeeping():
    """
    Loads the best results from all experiment run-throughs and stores them
    together in on variable

    Outputs:
        final_results: Array of the best results from all experiment
        run-throughs
    """
    files = [os.path.join(current_dir, 'model', str(f), 'best_scores.pickle')
             for f in range(1, exp_runthrough+1)]
    final_results = np.zeros((exp_runthrough, 14))
    for f in range(len(files)):
        with open(files[f], 'rb') as file:
            data = pickle.load(file)
        current_model_dir = os.path.join(current_dir, 'model', str(f+1))
        if data[0] == 0:
            pass
        else:
            path = os.path.join(current_model_dir,
                                'md_'+str(data[-1])+'_epochs.pth')
            mod_paths = [os.path.join(current_model_dir, l) for l in os.listdir(
                current_model_dir) if 'md_' in l]
            del mod_paths[mod_paths.index(path)]

            for m in mod_paths:
                os.remove(m)
            final_results[f, :] = data[0:-1]

    return final_results


def train(model, workspace_files_dir):
    """
    Sets up the experiment and runs the experiment. The training batches are
    loaded and pushed to the model, the resuls are analysed until the next
    epoch. At this point, the validation set is run in the same manner.
    Results and outputs are collated and recorded for future analysis and
    checkpointing

    Input
        model: obj - The model to be used for experimentation
        workspace_files_dir: str - Location of the programme code
    """
    num_of_classes = len(config_dataset.LABELS)
    learning_rate = config.EXPERIMENT_DETAILS['LEARNING_RATE']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    main_logger.info(f"Optimiser: ADAM. Learning Rate: {learning_rate}")
    # Need to load the model output variables first to build generator below
    # After the generator is built, we can load any previous unfinished model
    # Loading all states and variables before the generator alters the random
    # number generator for continuing training and therefore not reproducible
    if checkpoint:
        df, comp_train_pred, comp_val_pred, best_scores, data_saver = \
            util.load_model_outputs(model_dir)
    else:
        # train_acc, train_fscore, train_loss, val_acc, val_fscore, val_loss
        best_scores = [0] * 16
        comp_train_pred = comp_val_pred = 0
        df = pd.DataFrame(columns=config_dataset.COLUMN_NAMES)
        data_saver = {}
    print('Generating data for training')
    # train_info tuple of number of zeros, ones and class weights and if
    # gender balance is specified, gender weights (also tuple of
    # (fem_nd_w, fem_d_w, male_nd_w, male_d_w)
    gen, train_info = organiser.run_train(config,
                                          main_logger,
                                          checkpoint,
                                          features_dir,
                                          data_saver)
    if checkpoint:
        start_epoch = util.load_model(checkpoint_run,
                                      model,
                                      optimizer)
    else:
        start_epoch = 0
        # train_acc, train_fscore, train_loss, val_acc, val_fscore, val_loss

    avg_counter = per_epoch_train_pred = per_epoch_val_pred = 0
    # Train/Val, Accuracy, Precision, Recall, Fscore, Loss(single), mean_acc/f
    complete_results = np.zeros(30)

    gender_balance = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']
    cw = train_info[-3]
    start_new_timer = False
    start_timer = time.time()
    print('Beginning Training')
    comp = []
    # batch is tuple of data, labels, epoch and reset
    for (iteration, (batch, train_batch_loc, data_saver)) in enumerate(
            gen.generate_train_data(start_epoch)):
        if start_new_timer:
            start_timer = time.time()
            start_new_timer = False
        avg_counter += 1
        model.train()
        batch_labels = torch.LongTensor(batch[1])
        batch_output = get_output_from_model(model=model,
                                             data=batch[0])
        batch_folders = batch[2]
        comp.append(batch_folders)

        batch_weights = find_batch_weights(batch_folders,
                                           cw)

        train_loss = mu.calculate_loss(batch_output,
                                       batch_labels,
                                       batch_weights,
                                       gender_balance)

        if gender_balance:
            batch_labels = batch_labels % 2

        # Zero the gradient and Backprop the loss through the network
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        init = initialiser(avg_counter)
        complete_results[0:15], per_epoch_train_pred = \
            prediction_and_accuracy(batch_output,
                                    batch_labels,
                                    init,
                                    num_of_classes,
                                    complete_results[0:15],
                                    train_loss,
                                    per_epoch_train_pred)

        begin_evaluate = begin_evaluation(analysis_mode,
                                          batch[-2],
                                          batch[-1],
                                          iteration,
                                          iteration_epoch)
        if begin_evaluate:
            if analysis_mode == 'epoch':
                placeholder = batch[-2]
            else:
                placeholder = iteration_epoch // learn_rate_factor

            if validate:
                start_new_timer = True
                calculate_time(start_timer,
                               time.time(),
                               'train',
                               main_logger,
                               placeholder,
                               iteration)
                print('Evaluating - Development at epoch: ', placeholder)
                (complete_results[15:], per_epoch_val_pred) = evaluate(
                    model=model,
                    generator=gen,
                    data_type='dev',
                    class_weights=train_info[-1],
                    comp_res=complete_results[15:],
                    class_num=num_of_classes,
                    f_score_average=None,
                    epochs=start_epoch,
                    logger=main_logger,
                    gender_balance=gender_balance)

            complete_results, best_scores = \
                update_complete_results(complete_results,
                                        avg_counter,
                                        placeholder,
                                        best_scores)
            avg_counter = 0
            df.loc[placeholder-1] = complete_results

            complete_results = np.zeros(30)
            plot_graph(placeholder,
                       df,
                       final_iteration,
                       model_dir,
                       vis=vis, val=validate)

            comp_train_pred, comp_val_pred = compile_train_val_pred(
                per_epoch_train_pred,
                per_epoch_val_pred,
                comp_train_pred,
                comp_val_pred,
                placeholder)

            # Reduce learning rate
            if placeholder % learn_rate_factor == 0:
                reduce_learning_rate(optimizer)

            # Save model
            data_saver['class_weights'] = cw
            util.save_model(placeholder,
                            model,
                            optimizer,
                            main_logger,
                            model_dir)
            util.save_model_outputs(model_dir,
                                    df,
                                    comp_train_pred,
                                    comp_val_pred,
                                    best_scores,
                                    data_saver)

            # Stop learning
            patience = final_iteration+1
            if placeholder % patience == 0:
                reference = df['val_loss'].tolist()[-patience]
                trial = df['val_loss'].tolist()[-1]
                if not trial < reference:
                    print(f"Val_loss - Patience {reference}\nVal loss - "
                          f"Current {trial}")
                    final_organisation(best_scores,
                                       comp_train_pred,
                                       comp_val_pred,
                                       df,
                                       patience,
                                       placeholder,
                                       workspace_files_dir,
                                       data_saver)

                    plot_graph(placeholder,
                               df,
                               final_iteration,
                               model_dir,
                               early_stopper=True,
                               vis=vis,
                               val=validate)
                    break
            elif placeholder == final_iteration:
                final_organisation(best_scores,
                                   comp_train_pred,
                                   comp_val_pred,
                                   df,
                                   patience,
                                   placeholder,
                                   workspace_files_dir,
                                   data_saver)
                break


def test():
    """
    Re-runs experiment details from config file by loading the best
    performing epochs and if data_type='dev' re-run using the validation
    data, if data_type='test' re-run using the test data. The test set
    results will be displayed according to the user specified argument
    prediction_metric. There are 3 options, pick the best experiment model,
    average the experiment models, calculate the majority vote of the
    experiment models.
    """
    if validate:
        tester = False
        data_type = 'dev'
    else:
        tester = True
        data_type = 'test'

    counter = 0
    if config.EXPERIMENT_DETAILS['SPLIT_BY_GENDER']:
        comp_scores = np.zeros((exp_runthrough, 20))
    else:
        comp_scores = np.zeros((exp_runthrough, 10))

    if data_type == 'test':
        results = {'output': {}, 'target': {}, 'accum': {}}
        if config.EXPERIMENT_DETAILS['SPLIT_BY_GENDER']:
            results = {'female': {'output': {}, 'target': {}, 'accum': {}},
                       'male': {'output': {}, 'target': {}, 'accum': {}}}

    for exp_num in range(exp_runthrough):
        model_dir = os.path.join(current_dir,
                                 'model',
                                 folder_extensions[exp_num])
        if data_type == 'test':
            path_to_logger_for_test = os.path.join(features_dir,
                                                   sub_dir + '_test')
        else:
            path_to_logger_for_test = None

        if data_type == 'test' and counter == 0 or data_type == 'dev':
            main_logger, model, _, _, _ = setup(current_dir,
                                                model_dir,
                                                data_type,
                                                path_to_logger_for_test)
        num_of_classes = len(config_dataset.LABELS)

        optimizer = torch.optim.Adam(model.parameters())
        gender_balance = config.EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']

        for file in os.listdir(model_dir):
            if file.endswith(".pth"):
                current_epoch = int(file.split('_')[1])
                model_dir = os.path.join(model_dir, file)

        _ = util.load_model(checkpoint_path=model_dir,
                            model=model,
                            optimizer=optimizer)
        data_saver = util.load_model_outputs(model_dir,
                                             'test')

        if data_type == 'test' and counter == 0 or data_type == 'dev':
            generator, cw = organiser.run_test(config,
                                               main_logger,
                                               False,
                                               features_dir,
                                               data_saver,
                                               tester)
        f_score = None
        if config.EXPERIMENT_DETAILS['SPLIT_BY_GENDER']:
            start = 0
            for gen_number, gen in enumerate(generator):
                scores, per_epoch = evaluate(model,
                                             gen,
                                             data_type,
                                             cw[-3],
                                             np.zeros(15),
                                             num_of_classes,
                                             f_score,
                                             current_epoch,
                                             main_logger,
                                             gender_balance)

                scores[8] = np.mean(scores[0:2])
                scores[9] = np.mean(scores[6:8])
                scores = [scores[8], scores[0], scores[1], scores[9],
                          scores[6], scores[7], scores[11], scores[12],
                          scores[13], scores[14]]
                if gen_number == 0:
                    dictionary_key = 'female'
                else:
                    dictionary_key = 'male'

                print(f"{dictionary_key} Scores: {scores}")
                main_logger.info(f"{dictionary_key} Scores are: \n{scores}")
                comp_scores[exp_num, start:start + 10] = scores
                start += 10

                if data_type == 'test':
                    per_epoch, outputs, accum, targets = per_epoch
                    for count, folder in enumerate(outputs):
                        if folder not in results[dictionary_key]['output'].keys():
                            results[dictionary_key]['output'][folder] = np.array(outputs[folder][0])
                            results[dictionary_key]['accum'][folder] = accum[folder]
                            if gender_balance:
                                results[dictionary_key]['target'][folder] = \
                                    targets[folder] % 2
                            else:
                                results[dictionary_key]['target'][folder] = \
                                    targets[folder]
                        else:
                            results[dictionary_key]['output'][folder] = np.append(
                                results[dictionary_key]['output'][folder],
                                outputs[folder][0])
        else:
            scores, per_epoch = evaluate(model,
                                         generator,
                                         data_type,
                                         cw[-3],
                                         np.zeros(15),
                                         num_of_classes,
                                         f_score,
                                         current_epoch,
                                         main_logger,
                                         gender_balance)

            scores[8] = np.mean(scores[0:2])
            scores[9] = np.mean(scores[6:8])
            scores = [scores[8], scores[0], scores[1], scores[9],
                      scores[6], scores[7], scores[11], scores[12],
                      scores[13], scores[14]]

            print("Scores: ", scores)
            comp_scores[exp_num, :] = scores
            main_logger.info(f"Scores are: \n{scores}")

            if data_type == 'test':
                per_epoch, outputs, accum, targets = per_epoch
                for count, folder in enumerate(outputs):
                    if folder not in results['output'].keys():
                        results['output'][folder] = np.array(outputs[folder][0])
                        results['accum'][folder] = accum[folder]
                        if gender_balance:
                            results['target'][folder] = targets[folder] % 2
                        else:
                            results['target'][folder] = targets[folder]
                    else:
                        results['output'][folder] = np.append(
                            results['output'][folder], outputs[folder][0])

    if data_type == 'test':
        if config.EXPERIMENT_DETAILS['SPLIT_BY_GENDER']:
            final_results = {'female': {}, 'male': {}}
            comp_scores = []
            for gen_ind in final_results:
                temp_scores = evaluation_for_test(results[gen_ind],
                                                  num_of_classes,
                                                  f_score,
                                                  main_logger)
                comp_scores = comp_scores + temp_scores
                print(f"{gen_ind} Test output Scores: {temp_scores}")
                main_logger.info(f"{gen_ind} Test output Scores:"
                                 f" {temp_scores}")
        else:
            comp_scores = evaluation_for_test(results,
                                              num_of_classes,
                                              f_score,
                                              main_logger)
            print("Test output Scores: ", comp_scores)
            main_logger.info(f"Test output Scores: {comp_scores}")
    else:
        # Acc_avg, Acc_0, Acc_1, F_avg, F_0, F_1, tn_fp_fn_tp
        comp_scores_avg = np.mean(comp_scores, axis=0)
        print(f"\nAverage: {comp_scores_avg}")
        main_logger.info(f"Average Scores: \n{comp_scores_avg}")
    if data_type == 'dev' and config.EXPERIMENT_DETAILS['SPLIT_BY_GENDER']:
        tp1 = np.sum(comp_scores[:, 9])
        fn1 = np.sum(comp_scores[:, 8])
        fp1 = np.sum(comp_scores[:, 7])
        tn1 = np.sum(comp_scores[:, 6])
        tp2 = np.sum(comp_scores[:, -1])
        fn2 = np.sum(comp_scores[:, -2])
        fp2 = np.sum(comp_scores[:, -3])
        tn2 = np.sum(comp_scores[:, -4])
        matrix1 = np.array([['', ' ', '    Prediction', ''],
                            ['-', '-', '      ND', '    D '],
                            ['Ground', 'ND', tn1, fp1],
                            ['Truth', ' D ', fn1, tp1]])
        matrix2 = np.array([['', ' ', '    Prediction', ''],
                            ['-', '-', '      ND', '    D '],
                            ['Ground', 'ND', tn2, fp2],
                            ['Truth', ' D ', fn2, tp2]])
        print('Female Not Normalised:')
        print(matrix1)
        print('Male Not Normalised:')
        print(matrix2)
        main_logger.info(f"Female and Male Confusion Matrices")
        main_logger.info(matrix1)
        main_logger.info(matrix2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='mode')

    parser_train = sub_parser.add_parser('train')
    parser_train.add_argument('--validate',
                              action='store_true',
                              default=False,
                              help='set whether we want to use a validation '
                                   'set')
    parser_train.add_argument('--vis',
                              action='store_true',
                              default=False,
                              help='determine whether model graph is output')
    parser_train.add_argument('--cuda',
                              action='store_true',
                              default=False,
                              help='pass --cuda if you want to run on GPU')
    parser_train.add_argument('--debug',
                              action='store_true',
                              default=False,
                              help='set the program to run in debug mode '
                                   'means, that the most recent folder will '
                                   'be deleted automatically to speed up '
                                   'debugging')
    parser_train.add_argument('--position',
                              type=int,
                              default=1,
                              help='Used to determine which main and config '
                                   'files to save')
    parser_train.add_argument('--limited_memory',
                              action='store_true',
                              default=False,
                              help='Set to true if working with less than '
                                   '32GB RAM')
    parser_train.add_argument('--prediction_metric',
                              type=int,
                              default=0,
                              help='NOT APPLICABLE FOR TRAINING')

    parser_test = sub_parser.add_parser('test')
    parser_test.add_argument('--vis',
                             action='store_true',
                             default=False,
                             help='determine whether model graph is output')
    parser_test.add_argument('--cuda',
                             action='store_true',
                             default=False,
                             help='pass --cuda if you want to run on GPU')
    parser_test.add_argument('--validate',
                             action='store_true',
                             default=False,
                             help='Do you want to run the respective '
                                  'validation folds used during training in '
                                  'testing?')
    parser_test.add_argument('--position',
                             type=int,
                             default=1,
                             help='Used to determine which main and config '
                                  'files to save')
    parser_test.add_argument('--prediction_metric',
                             type=int,
                             default=0,
                             help='How do you want to average the outputs of '
                                  'all models? 0: best single model, '
                                  '1: average of models, 2: majority vote of '
                                  'models')
    parser_test.add_argument('--debug',
                             action='store_true',
                             default=False,
                             help='set the program to run in debug mode '
                                  'means, that the most recent folder will be '
                                  'deleted automatically to speed up debugging')
    args = parser.parse_args()

    mode = args.mode
    debug = args.debug
    vis = args.vis
    cuda = args.cuda
    validate = args.validate
    position = args.position
    prediction_metric = args.prediction_metric
    workspace_main_dir = config.WORKSPACE_MAIN_DIR
    features_dir = os.path.join(workspace_main_dir, config.FOLDER_NAME)
    gender = config.GENDER
    if gender == 'm' or gender == 'f':
        features_dir = features_dir + '_gen'
    print('feature_dir:', features_dir)
    chosen_seed = config.EXPERIMENT_DETAILS['SEED']
    analysis_mode = config.ANALYSIS_MODE

    iteration_epoch = config.EXPERIMENT_DETAILS['ITERATION_EPOCH']
    if analysis_mode == 'epoch':
        final_iteration = config.EXPERIMENT_DETAILS['TOTAL_EPOCHS']
    elif analysis_mode == 'iteration':
        final_iteration = config.EXPERIMENT_DETAILS['TOTAL_ITERATIONS']
        final_iteration = round(final_iteration / iteration_epoch)

    exp_runthrough = config.EXPERIMENT_DETAILS['EXP_RUNTHROUGH']
    folder_extensions = [str(i) for i in range(1, exp_runthrough+1)]

    sub_dir = config.EXPERIMENT_DETAILS['SUB_DIR']
    current_dir = os.path.join(features_dir, sub_dir)
    total_score = np.zeros((exp_runthrough, 14))
    if args.mode == 'train':
        for i in range(exp_runthrough):
            model_dir = os.path.join(current_dir, 'model', folder_extensions[i])
            main_logger, model, checkpoint_run, checkpoint, next_exp = setup(
                current_dir,
                model_dir)
            if next_exp:
                break
            comp_start_time = time.time()
            train(model,
                  config.WORKSPACE_FILES_DIR)
            if i+1 == exp_runthrough:
                total_score = bookeeping()
                mean_score = np.mean(total_score, axis=0)
                for s in total_score:
                    main_logger.info(list(s))
                main_logger.info(list(mean_score))
                print(list(mean_score))
            comp_end_time = time.time()
            complete_time = comp_end_time - comp_start_time
            main_logger.info(f"Complete time to run model: {complete_time}")
            handlers = main_logger.handlers[:]
            for handler in handlers:
                handler.close()
                main_logger.removeHandler(handler)
            chosen_seed += 100
    elif args.mode == 'test':
        test()
    else:
        raise Exception('There has been an error in the input arguments')
