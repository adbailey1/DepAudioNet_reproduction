**Prerequisites**

Install miniconda and load the environment file from environment.yml file

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

**Notes**

So far the audio and textual data have been experimented with with the
 ability to process the visual data to be conducted in the future.  

