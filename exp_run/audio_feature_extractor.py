import numpy as np
import librosa


class LogMelExtractor:
    """
    Creates a log-Mel Spectrogram of some input audio data. It first creates
    a mel filter and then applies the transformation of this mel filter to
    the STFT representation of the audio data

    Inputs
        sample_rate: int - The sampling rate of the original audio data
        window_size: int - The size of the window to be used for the mel
                     filter and the STFT transformation
        hop_size: int - The distance the window function will move over the
                  audio data - Related to the overlap = window_size - hop_size
        mel_bins: int - The number of bins for the mel filter
        fmin: int - The minimum frequency to start working from default=0
        fmax: int - The maximum frequency to start working from. Nyquist limit

    Output
        logmel_spectrogram: numpy.array - The log-Mel spectrogram
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax):
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)

        # Output in the form of ((n_fft//2 + 1), mel_bins)
        self.melW = librosa.filters.mel(sr=sample_rate,
                                        n_fft=window_size,
                                        n_mels=mel_bins,
                                        fmin=fmin,
                                        fmax=fmax)

    def transform(self, audio):
        """
        Performs the transformation of the mel filter and the STFT
        representation of the audio data
        """
        # Compute short-time Fourier transform
        # Output in the form of (N, (n_fft//2 + 1))
        stft_matrix = sepctrogram(audio=audio,
                                  window_size=self.window_size,
                                  hop_size=self.hop_size,
                                  squared=True,
                                  window_func=self.window_func)

        # Mel spectrogram
        mel_spectrogram = np.dot(stft_matrix.T, self.melW.T)

        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10,
            top_db=None)

        logmel_spectrogram = logmel_spectrogram.astype(np.float32).T

        return logmel_spectrogram


def sepctrogram(audio, window_size, hop_size, squared,
                window_func=np.hanning(1024)):
    """
    Computes the STFT of some audio data.

    Inputs
        audio: numpy.array - The audio data
        window_size: int - The size of the window passed over the data
        hop_size: int - The distance between windows
        squared: bool - If True, square the output matrix
        window_func: numpy.array - The window function to be passed over data
    """
    stft_matrix = librosa.core.stft(y=audio,
                                    n_fft=window_size,
                                    hop_length=hop_size,
                                    window=window_func,
                                    center=True,
                                    dtype=np.complex64,
                                    pad_mode='reflect')

    stft_matrix = np.abs(stft_matrix)
    if squared:
        stft_matrix = stft_matrix ** 2

    return stft_matrix


def create_mfcc_delta(feature, concat=False):
    """
    Obtains the local differential (first and second order) of the MFCC

    Inputs
        feature: np.array - The MFCC to be used for the local differentials
        concat: bool - If True, the differentials will be concatenated rather
                than stacked

    Output
        mfcc: numpy.array - The Updated MFCC
    """
    mfcc_delta = librosa.feature.delta(feature)
    mfcc_delta2 = librosa.feature.delta(feature, order=2)

    if concat:
        mfcc = np.concatenate((feature, mfcc_delta, mfcc_delta2))
    else:
        mfcc = np.array((feature, mfcc_delta, mfcc_delta2))

    return mfcc


def create_delta(feature, n_total=2, delta_order=2):
    """
    Creates local differentials by time shifting the data (delay and advance)
    with variable regression windows (N default is 2) according to the
    formula found in "Learning Affective Features With a Hybrid Deep
    Model for Audio–Visual Emotion Recognition" Zhang et al. 2017 IEEE
    Transactions on Circuit and Systems for Video Technology Vol. 28 No. 10
    d_t = Sum_(n=1)^N n(c_(t+n) - c_(t-n)) / 2* Sum_(n=1)^N n^2

    Inputs
        feature: numpy.array - The feature array used to calculate differentials
        n_total: int - The length of the regression window
        delta_order: int - The number of differentials to calculate

    Output
        feature: numpy.array - The updated features with their differentials
    """
    rows = feature.shape[-2]
    columns = feature.shape[-1]

    def differences(feat, n):
        a = np.zeros((rows, columns))
        b = np.zeros((rows, columns))
        a[:, n:] = feat[:, :-n]
        b[:, 0:-n] = feat[:, n:]
        delta_diff = b - a

        return delta_diff * n

    if len(feature.shape) < 3:
        feature = np.reshape(feature, (1, rows, columns))
    while True:
        n_out = delta = 0
        for j in range(n_total, 0, -1):
            delta_new = differences(feature[0, :, :], j)
            delta = delta_new - delta
            n_out += j ** 2
        delta = delta / (n_out*2)
        delta = np.reshape(delta, (1, rows, columns))
        if delta_order == 1:
            return np.vstack((feature, delta))
        else:
            return np.vstack((feature, create_delta(delta, n_total,
                                                    delta_order-1)))


def feature_segmenter(feature, meta, feature_exp, dim, convert_to_image=False):
    """
    Segments the features into dimensions specified by feature.shape[-1] and
    dim. The number of extra dimensions is used to create lists of the
    folder, class, score and index for this updated reshaped data array

    Inputs:
        feature: The feature array to be segmented
        meta: Includes Folder, Class, Score, and Gender
        feature_exp: Type of feature experiment eg. logmel
        dim: Value to segment the data by
        convert_to_image: Bool - Is the feature array being converted to 3D?

    Outputs:
        new_features: Updated array of features N, F, S where S is the
                      feature dimension specified in the config file.
        new_folders: Updated list of folders
        new_classes: Updated list of classes
        new_scores: Updated list of scores
        new_indexes: Updated list of indexes
    """
    if feature.shape[1] % dim == 0:
        num_extra_dimensions = (feature.shape[1] // dim)
    else:
        num_extra_dimensions = (feature.shape[1] // dim) + 1
    new_indexes = np.arange(num_extra_dimensions)
    if convert_to_image:
        if feature_exp == 'MFCC_concat':
            new_features = np.zeros([num_extra_dimensions, feature.shape[
                0]*3, dim], dtype=np.float32)
        else:
            new_features = np.zeros([num_extra_dimensions, 3, feature.shape[0],
                                     dim],
                                    dtype=np.float32)
    else:
        new_features = np.zeros([num_extra_dimensions, feature.shape[0], dim],
                                dtype=np.float32)

    # E.g. if mel bins = 128 and segments = 512, and original length of
    # data = 8236, goes through all features and takes segments of 512
    # samples to make a new variable with dimensions of (x, 128,
    # 512) instead of (x, 128, 8236) 512 doesn't exactly fit into 8236/512
    # therefore the last section will be zero padded. No overlap has been
    # considered so far.
    last_dim = feature.shape[-1]
    leftover = dim - (last_dim % dim)
    if convert_to_image:
        if feature_exp == 'MFCC' or feature_exp == 'MFCC_concat':
            feature = create_mfcc_delta(feature)
        else:
            feature = create_delta(feature)
        if leftover != dim:
            if feature_exp == 'MFCC_concat':
                feature = np.hstack((feature, np.zeros((feature.shape[0],
                                                        leftover))))
            else:
                feature = np.dstack((feature, np.zeros((feature.shape[0],
                                                        feature.shape[1],
                                                        leftover))))
        if feature_exp == 'MFCC_concat':
            new_features[:, :, :] = np.split(feature, num_extra_dimensions,
                                             axis=1)
        else:
            new_features[:, :, :, :] = np.split(feature, num_extra_dimensions,
                                                axis=2)
    else:
        if leftover != dim:
            feature = np.hstack((feature, np.zeros((feature.shape[0],
                                                    leftover))))
        new_features[:, :, :] = np.split(feature, num_extra_dimensions,
                                         axis=1)

    new_folders = [meta[0]] * num_extra_dimensions
    new_classes = [meta[1]] * num_extra_dimensions
    new_scores = [meta[2]] * num_extra_dimensions
    new_gender = [meta[3]] * num_extra_dimensions

    return (new_features, new_folders, new_classes, new_scores, new_gender,
            new_indexes)


def moving_average(data, N, decimation=False):
    """
    Creates a moving average filter and applies it to some input data

    Inputs:
        data: numpy.array - The data to be filtered
        N: int - The size of the filter
        decimation: bool - Set True downsamples the data

    Output
        ma_data: numpy.array - The filtered input data
    """
    average_mask = np.ones(N) / N
    if decimation:
        ma_data = np.convolve(data, average_mask, 'full')
        return ma_data[N-1::N]
    else:
        ma_data = np.convolve(data, average_mask, 'same')
        return ma_data
