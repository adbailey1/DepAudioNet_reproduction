**Credit**

This repository relates to our work in the EUSIPCO 2021 paper, "Gender Bias in Depression Detection Using Audio Features", https://arxiv.org/abs/2010.15120

**Prerequisites**

This was develpoed for Ubuntu (18.04) with Python 3

Install miniconda and load the environment file from environment.yml file

`conda env create -f environment.yml`

(Or should you prefer, use the text file)

Activate the new environment: `conda activate myenv`

**Dataset**

For this experiment, the DAIC-WOZ dataset is used (found in AVEC16 - AVEC17, 
not the newer extended version). This can be 
obtained
 through The University of Southern California (http://dcapswoz.ict.usc.edu
 /) by signing an agreement form. The dataset is roughly 135GB. 
 
 The dataset contains many errors and noise (such as interruptions during an
  interview or missing transcript files for the virtual agent). It is
   recommended to download and run my DAIC-WOZ Pre-Processing Framework in
    order to quickly begin experimenting with this data (https://github.com/adbailey1/daic_woz_process)

**Experiment Setup**

Use the config file to set experiment preferences and locations of the code, workspace, and dataset directories. There are two config files here. config.py is usually used as a template and further config files are added with the suffix '_1', '_2' etc for different experiments. 

Updated the run.sh file if you want to run the experiment through bash (call
 `./run.sh` from terminal). The arguments required by calling main1.py
  are: 
 - `train` - to train a model 
- `test` - to test a trained model

Optional commands are: 
- `--validate` - to train a model with a validation set
- `--cuda` - to train a model using a GPU
- `--vis` - to visualise the learning graph at the end of every epoch
- `--position` - to specify which main1.py and config.py file are used for this 
  experiment  
- `--debug` - for debug mode which automatically overwrites an previous data at
  a directory for quicker debugging.
  
TEST MODE ONLY:
-  `--prediction_metric` - this determines how the output is calculated 
   running on the test set in test mode. 0 = best performing model, 1 = 
   average of all 
   models, 2 = majority vote from all models  

For example: To run a training experiment without bash, using a validation
 set, GPU, not visualising the per epoch results graphs, and using main1.py 
and config_1.py files:
 
 `python3 main1.py train --validate --cuda --vis --position=1`
 
 Without using the validation set:
 
 `python3 main1.py train --cuda --vis --position=1`
 
 Running trained models again on the validation set:
 
 `python3 main1.py test --validate --cuda --vis --position=1`
 
 Running trained models on the test set:
 
 `python3 main1.py test --cuda --vis --position=1`

**Results Mel-Spectrogram**

All results (calculated on the validation set) are found in our paper: 
https://arxiv.org/abs/2010.15120

We replicated DepAduioNet's reported results by using the following config settings:
````
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
                      'CLASS_WEIGHTS': False,
                      'USE_GENDER_WEIGHTS': False,
                      'SUB_SAMPLE_ND_CLASS': True,  # Make len(dep) == len(
                      # ndep)
                      'CROP': True,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': True,  # Only for use in test mode
                      'FEATURE_DIMENSIONS': 120,
                      'FREQ_BINS': 40,
                      'BATCH_SIZE': 20,
                      'SVN': True,
                      'LEARNING_RATE': 1e-3,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': 'exp_1',
                      'EXP_RUNTHROUGH': 5}
# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = True
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'
````

NOTE: The mel-spectrogram needs to be computed following Ma et al. {DepAudioNet: An Efficient Deep Model for Audio based Depression Classification (https://dl.acm.org/doi/10.1145/2988257.2988267)} procedure: calculated mel spectrogram per file and calculate: (file - mean) / standard deviation. Check out https://github.com/adbailey1/daic_woz_process for a pre-processing framework that should handle the feature database creation.

Import/Global values in main1.py:

- Use: `from exp_run.models_pytorch import CustomMel7 as CustomMel`
- Use: `learn_rate_factor = 2`

Results on Dataset's Validation Set: 

````
|F1(ND)|F1(D) |F1 avg|
| .732 | .522 | .627 |
````

- Use: `learn_rate_factor = 3`

Results on Dataset's Validation Set: 

````
|F1(ND)|F1(D) |F1 avg|
| .740 | .539 | .634 | 
````

**Raw Audio Results**
In config1.py:

````
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'raw',
                      'CLASS_WEIGHTS': False,
                      'USE_GENDER_WEIGHTS': False,
                      'SUB_SAMPLE_ND_CLASS': True,  # Make len(dep) == len(
                      # ndep)
                      'CROP': True,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': True,  # Only for use in test mode
                      'FEATURE_DIMENSIONS': 61440,
                      'FREQ_BINS': 1,
                      'BATCH_SIZE': 20,
                      'SVN': True,
                      'LEARNING_RATE': 1e-3,
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'SUB_DIR': 'exp_1',
                      'EXP_RUNTHROUGH': 5}
# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = True
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'
````

NOTE: The raw audio needs to be computed in the same way as the mel spectrogram according to: (file - mean) / standard deviation. Check out https://github.com/adbailey1/daic_woz_process for a pre-processing framework that should handle the feature database creation.

Import/Global values in main1.py:

- Use: `from exp_run.models_pytorch import CustomRaw3 as CustomRaw`
- Use: `learn_rate_factor = 2`

Results on Dataset's Validation Set: 

````
|F1(ND)|F1(D) |F1 avg|
| .766 | .531 | .648 |
````

- Use: `learn_rate_factor = 3`

Results on Dataset's Validation Set:

````
|F1(ND)|F1(D) |F1 avg|
| .796 | .520 | .658 | 
````

**Notes**

So far the audio data have been experimented with. The
 ability to process the textual and/or visual data to be conducted in the 
future.  

