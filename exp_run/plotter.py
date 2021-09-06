import matplotlib.pyplot as plt
import librosa
from librosa import display
import os
import numpy as np
import pandas as pd
from utilities.utilities_main import create_directories


def mel(mel_filter):
    """
    Plots a mel filter

    Input
        mel_filter: numpy.array - The mel filter to be plotted
    """
    plt.figure()
    librosa.display.specshow(mel_filter, x_axis='linear')
    plt.ylabel('Mel Filter')
    plt.title('Mel Filter Bank')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def save_plain_plot(directory, folder_name, spectrogram, spec_type='logmel'):
    """
    Saves a basic image of a spectrogram with no axes. If the directory for
    saving the image does not exist, one will be created

    Inputs
        directory: str - Location to save the images of the spectrogram
        folder_name: str - Name of the file to be saved
        spectrogram: numpy.array - The spectrogram data to be saved
        spec_type: str - Used for folder name of the type of spectrogram to
                   be saved
    """
    save_path_images = os.path.join(directory, spec_type)
    if not os.path.exists(save_path_images):
        create_directories(directory, spec_type)

    save_path_images = os.path.join(save_path_images, folder_name)

    if not os.path.exists(save_path_images):
        f = plt.figure()
        librosa.display.specshow(spectrogram)
        plt.tight_layout()
        f.savefig(save_path_images)
        plt.close()


def plot_spectrogram(spec, type_of_spec=''):
    """
    Plots spectrogram or logmel spectrogram

    Inputs
        spec: numpy.array - The spectrogram to be plotted
        type_of_spec: str - Used to determine the type of spectrogram
    """
    if type_of_spec == '':
        librosa.display.specshow(spec, x_axis='frames')
    else:
        librosa.display.specshow(spec, x_axis='frames', y_axis=type_of_spec)
    plt.colorbar(format='%+2.0f dB')
    plt.title(type_of_spec + ' Spectrogram')
    plt.tight_layout()
    plt.show()


def plot_graph(epoch, results, total_epochs, model_dir, early_stopper=False,
               vis=False, val=True):
    """
    Plots a graph of the performance of a network up to a certain epoch

    Inputs
        epoch: int - The current epoch
        results: pandas.dataframe - Complete results of experiment
        total_epochs: int - The total number of epochs for the experiment -
                      determines when to save the plot
        model_dir: str - The location where the model is saved
        early_stopper: bool - If True, saves the plot even if total_epochs
                       has not been reached
        vis: bool - Set True if plot is to be visible after every epoch
        val: bool - Set True if training with a validation set

    Outputs:
        fig: matplot of the current progress graph
    """
    x_values = list(range(1, epoch+1))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    l1 = ax1.plot(x_values, results['train_loss'].tolist(), 'r.-',
                  label="trn_loss")
    l2 = ax2.plot(x_values, results['train_mean_fscore'].tolist(), 'm.-',
                  label="trn_F1")
    l3 = ax2.plot(x_values, results['train_mean_acc'].tolist(), 'g.-',
                  label="trn_acc")
    if val:
        l4 = ax1.plot(x_values, results['val_loss'].tolist(), 'b.-',
                      label="val_loss")
        l5 = ax2.plot(x_values, results['val_mean_fscore'].tolist(), 'k.-',
                      label="val_F1")
        l6 = ax2.plot(x_values, results['val_mean_acc'].to_list(), 'y.-',
                      label="val_acc")
        lns = l1 + l2 + l3 + l4 + l5 + l6
    else:
        lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    if not vis and not early_stopper and epoch % 20 == 0:
        plt.show()
    if epoch == total_epochs or early_stopper:
        fig.savefig(os.path.join(model_dir, 'accuracy_loss_plot.png'))
    plt.close('all')

    return fig


def confusion_mat(target, predict):
    """
    Produces a confusion matrix for the binary class classification problem

    Inputs
        target: numpy.array - The labels for the dataset
        predict: numpy.array - The output predictions from a model

    Output
        cm: pandas.dataframe - The confusion matrix
    """
    matrix = np.array([['-', '-', 0, 1],
                       ['Ground', 0, 0, 0],
                       ['Truth', 1, 0, 0]])

    for counter, value in enumerate(target):
        if value == 0 and int(predict[counter]) == value:
            matrix[1][2] = int(matrix[1][2]) + 1
        elif value == 0 and int(predict[counter]) != value:
            matrix[1][3] = int(matrix[1][3]) + 1
        elif value == 1 and int(predict[counter]) != value:
            matrix[2][2] = int(matrix[2][2]) + 1
        else:
            matrix[2][3] = int(matrix[2][3]) + 1

    cm = pd.DataFrame(matrix, columns=['-', '-', 'Predicted', 'Values'])
    return cm
