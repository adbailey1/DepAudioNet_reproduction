import os
import sys


# Set to complete to use all the data
# Set to sub to use training/dev sets only
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'logmel',
                      'AUDIO_MODE_IS_CONCAT_NOT_SHORTEN': True,
                      'MAKE_DATASET_EQUAL': False,
                      'CLASS_WEIGHTS': False,
                      'OVERSAMPLING': False,
                      'FEATURE_OVERLAP_PERCENT': 0,
                      'FEATURE_DIMENSIONS': 512,
                      'MEL_BINS': 40,
                      'BATCH_SIZE': 32,
                      'LEARNING_RATE': 1e-2,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'SUB_DIR': 'exp_3a',
                      'TOTAL_FOLDS': 4,
                      'MODE': 'sub',
                      'DATASET_IS_BACKGROUND': False,
                      'CONVERT_TO_IMAGE': True,
                      'NETWORK': 'custom'}
# (Channels), Kernel, Stride, Pad
# GRU - input_size - number of channels from last layer (as int), hidden_size
# (as int), bidirectional (as Bool)
NETWORK_PARAMS = {'CONV_1': [[1, 32], (3, 3), (1, 1), (1, 1)],
                  'CONV_2': [[32, 32], (3, 3), (1, 1), (1, 1)],
                  'POOL_1': [(2, 2), (2, 2), (0, 0)],
                  'DROP_1': 0.5,
                  'HIDDEN_1': [524288, 2],
                  'SOFTMAX_1': 1}

FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']
TRAIN_SPLIT = 0.85
DEV_SPLIT = 0.05
TEST_SPLIT = 0.1

WINDOW_SIZE = 1024
OVERLAP = 50
HOP_SIZE = WINDOW_SIZE - round(WINDOW_SIZE * (OVERLAP / 100))
FMIN = 0
SAMPLE_RATE = 16000
FMAX = SAMPLE_RATE / 2
REMOVE_BACKGROUND = True

if EXPERIMENT_DETAILS['AUDIO_MODE_IS_CONCAT_NOT_SHORTEN']:
    extension = 'concat'
else:
    extension = 'shorten'
if EXPERIMENT_DETAILS['FEATURE_OVERLAP_PERCENT'] > 0:
    overlap = 'O'
else:
    overlap = 'NO'
if EXPERIMENT_DETAILS['MAKE_DATASET_EQUAL']:
    data_eq = '_equalSet_'
else:
    data_eq = '_'
if EXPERIMENT_DETAILS['CLASS_WEIGHTS']:
    cw = 'CW_'
else:
    cw = 'NCW_'
if EXPERIMENT_DETAILS['OVERSAMPLING']:
    sampling = 'OS'     # oversampling
else:
    sampling = 'RS'     # random sampling/ normal

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['MEL_BINS'])}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and REMOVE_BACKGROUND:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['MEL_BINS'])}_WIN_" \
                      f"{str(WINDOW_SIZE)}_OVERLAP_{str(OVERLAP)}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not REMOVE_BACKGROUND:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['MEL_BINS'])}_with_backgnd_exp"
else:
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and REMOVE_BACKGROUND:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_exp"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not REMOVE_BACKGROUND:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_with_backgnd_exp"
EXP_NAME = f"{extension}_{overlap}{data_eq}{cw}{sampling}"

SECONDS_TO_SEGMENT = 0.032
SEPARATOR = '###############################################################'

MINI_DATA = None

label = {'Not_Depressed', 'Depressed'}

LABELS = {label: i for i, label in enumerate(label)}
INDEX = {i: label for i, label in enumerate(label)}

EXCLUDED_SESSIONS = [342, 394, 398, 460]

COLUMN_NAMES = ['train_acc_0', 'train_acc_1', 'train_precision_0',
                'train_precision_1', 'train_reccall_0', 'train_recall_1',
                'train_fscore_0', 'train_fscore_1', 'train_mean_acc',
                'train_mean_fscore', 'train_loss', 'train_true_negative',
                'train_false_positive', 'train_false_negative',
                'train_true_positive', 'val_acc_0', 'val_acc_1',
                'val_precision_0', 'val_precision_1', 'val_reccall_0',
                'val_recall_1', 'val_fscore_0', 'val_fscore_1',
                'val_mean_acc', 'val_mean_fscore', 'val_loss',
                'val_true_negative', 'val_false_positive',
                'val_false_negative', 'val_true_positive',
                ]

if sys.platform == 'win32':
    DATASET = os.path.join('C:', '\\Users', 'Andrew', 'OneDrive', 'DAIC-WOZ')
    WORKSPACE = os.path.join('C:', '\\Users', 'Andrew', 'OneDrive', 'Coding', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
elif sys.platform == 'linux' and not os.uname()[1] == 'andrew-ubuntu':
    DATASET = os.path.join('/vol/vssp/datasets/singlevideo01/DAIC-WOZ')
    # set the path of the workspace (where the code is)
    WORKSPACE_FILES_DIR = '/user/HS227/ab01814/pycharm_projects/daic_woz_2'
    # set the path of the workspace (where the models/output will be stored)
    WORKSPACE_MAIN_DIR = os.path.join('/vol/research/ab01814_res', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
elif os.uname()[1] == 'andrew-ubuntu':
    DATASET = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/datasets/DAIC-WOZ'
    WORKSPACE_MAIN_DIR = '/mnt/6663b3e6-a12f-49e8-b881-421cebf2f8c6/daic_woz_2'
    WORKSPACE_FILES_DIR = os.path.join('/home', 'andrew', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join(DATASET, 'DAIC-WOZ', 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
    FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
    COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
else:
    DATASET = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ')
    WORKSPACE = os.path.join('Users', 'andrewbailey', 'OneDrive', 'Coding', 'PycharmProjects', 'daic_woz_2')
    TRAIN_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'train_split_Depression_AVEC2017.csv')
    DEV_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'dev_split_Depression_AVEC2017.csv')
    TEST_SPLIT_PATH = os.path.join('Users', 'andrewbailey', 'OneDrive', 'DAIC-WOZ', 'test_split_Depression_AVEC2017.csv')
