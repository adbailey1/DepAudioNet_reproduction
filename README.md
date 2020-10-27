**Prerequisites**

This was created using Python 3.7, future versions of Python may not be compatible with the packages in the environment files. 

Install miniconda and load the environment file from environment_conda.yml file

`conda env create -f environment.yml`

(Or should you prefer, use the text file)

Activate the new environment: `conda activate myenv`

**Dataset**

For this experiment, the DAIC-WOZ dataset is used. This can be obtained
 through The University of Southern California (http://dcapswoz.ict.usc.edu
 /) by signing an agreement form. The dataset is roughly 135GB. 
 
 The dataset contains many errors and noise (such as interruptions during an
  interview or missing transcript files for the virtual agent). It is
   recommended to download and run my DAIC-WOZ Pre-Processing Framework in
    order to quickly begin experimenting with this data (LINK)

**Experiment Setup**

Use the config file to set experiment preferences and locations of the code, workspace, and dataset directories. There are two config files here. config.py is usually used as a template and further config files are added with the suffix '_1', '_2' etc for different experiments. 

Updated the run.sh file if you want to run the experiment through bash (call
 ./run.sh from terminal). The arguments required by calling main.py
  are: 
 - train - to train a model 
- test - to test a trained model

Optional commands are: 
- --validate - to train a model with a validation set
- --cuda - to train a model using a GPU
- --vis - to visualise the learning graph at the end of every epoch
- --debug - for debug mode which automatically overwrites an previous data at
  a directory for quicker debugging. 

For example: To run a training experiment without bash, using a validation
 set, GPU, and not visualising the per epoch results graphs
 
 `python3 main.py train --validate --cuda --server`
 
**Results Mel**

We replicated DepAduioNet's reported results by using the following config settings:
`EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
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
ANALYSIS_MODE = 'epoch'`

NOTE: The mel-spectrogram needs to be computed following Ma et al. {DepAudioNet} procedure: calculated mel spectrogram per file and calculate: (file - mean) / standard deviation

Use "CustomMel7" for training.

Results:  Learning Rate update=2 
|F1(ND)|F1(D) |F1 avg|
| .725 | .520 | .622 |

Results:  Learning Rate update=3
|F1(ND)|F1(D) |F1 avg|
| .750 | .511 | .631 | 


**Results Raw**

We replicated DepAduioNet's reported results by using the following config settings:
`EXPERIMENT_DETAILS = {'FEATURE_EXP': 'raw',
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
ANALYSIS_MODE = 'epoch'`

NOTE: The raw audio needs to be computed in the same way as the mel spectrogram according to: (file - mean) / standard deviation

Use "CustomRaw3" for training.

Results:  Learning Rate update=2 
|F1(ND)|F1(D) |F1 avg|
| .738 | .510 | .624 |

Results:  Learning Rate update=3
|F1(ND)|F1(D) |F1 avg|
| .765 | .568 | .667 | 

**Notes**

So far the audio and textual data have been experimented with with the
 ability to process the visual data to be conducted in the future.  

