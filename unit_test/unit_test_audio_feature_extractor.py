from exp_run import audio_feature_extractor
import time
import numpy as np
import librosa


def segmenter_test(dimensions, overlap, config):
    """
    This test checks to see if the function can split the log-mel spectrogram
    into a specific number of segments
    :return:
    """
    audio_file = np.random.randn(10000000)
    sample_rate = config.SAMPLE_RATE
    window_size = 1024
    hop_size = 512
    f_min = config.FMIN
    f_max = config.FMAX
    mel_bins = config.EXPERIMENT_DETAILS['MEL_BINS']
    window_function = np.hanning(window_size)

    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=window_size,
                                     n_mels=mel_bins, fmin=f_min, fmax=f_max)
    stft_matrix = librosa.core.stft(y=audio_file,
                                    n_fft=window_size,
                                    hop_length=hop_size,
                                    window=window_function,
                                    center=True,
                                    dtype=np.complex64,
                                    pad_mode='reflect')
    spec_test = np.dot(mel_filter, np.abs(stft_matrix) ** 2)
    log_spec_test = librosa.core.power_to_db(spec_test, ref=1.0, amin=1e-10,
                                             top_db=None)
    log_spec_test = log_spec_test.astype(np.float32)
    if log_spec_test.shape[1] % dimensions == 0 and overlap == 0:
        expected_dimensions = log_spec_test.shape[1] // dimensions
    elif log_spec_test.shape[1] % dimensions == 0 and overlap > 0:
        hop = int((overlap / 100) * dimensions)
        expected_dimensions = ((log_spec_test.shape[1] - dimensions) // hop) + 1
    elif log_spec_test.shape[1] % dimensions != 0 and overlap == 0:
        expected_dimensions = (log_spec_test.shape[1] // dimensions) + 1
    else:
        hop = int((overlap / 100) * dimensions)
        expected_dimensions = ((log_spec_test.shape[1] - dimensions) // hop) + 2

    expected_dimensions = [expected_dimensions, mel_bins, dimensions]

    new_feat, _, _, _ = audio_feature_extractor.logmel_segmenter(
        log_spec_test, 300, 1, dimensions, overlap)

    x, y, z = new_feat.shape
    if x in expected_dimensions and y in expected_dimensions and z in \
            expected_dimensions:
        print('Pass: Expected Dimensions ', expected_dimensions,
              ' Dimensions Received ', [x, y, z])
    else:
        print('Test Failed')
        return False

    return True


def logmel_transform(window_size, hop_size, config):
    """
    Test to see determine whether the mel filter, STFT matrix and eventual
    log-mel spectrogram have the correct dimensions and values.
    :param window_size:
    :param hop_size:
    :return:
    """
    audio_file = np.random.randn(102400)
    sample_rate = config.SAMPLE_RATE
    f_min = config.FMIN
    f_max = config.FMAX
    mel_bins = config.EXPERIMENT_DETAILS['MEL_BINS']
    window_function = np.hanning(window_size)

    # number_of_fft=1024, due to the nyquist theorem, this will equate to N/2
    # in the frequency domain. Therefore for this example we expect the
    # frequency domain to be N/2+1
    expected_mel_shape_1 = mel_bins
    expected_mel_shape_2 = int((window_size / 2) + 1)

    mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=window_size,
                                     n_mels=mel_bins, fmin=f_min, fmax=f_max)

    if expected_mel_shape_1 in mel_filter.shape and expected_mel_shape_2 in \
            mel_filter.shape:
        print(f"Pass: Expected Mel Filter dimensions "
              f"{expected_mel_shape_1, expected_mel_shape_2}, Received: "
              f"{mel_filter.shape}")
    else:
        print('Mel Filter dimensions not equal')
        return False

    # stft_matrix dimensions should be frequency bins vs samples. Frequency
    # bins are the same as the mel_filter n_fft/2 + 1. Samples are
    # calculated: (audio_file_length - window_size/2) // hop_size
    # plus 1 for the initial window, 0 centred, and then effectively
    # zero_padded depending on the leftover
    expected_dimensions = ((audio_file.shape[0] - window_size/2) // hop_size)\
                          + 2
    stft_matrix = librosa.core.stft(y=audio_file,
                                    n_fft=window_size,
                                    hop_length=hop_size,
                                    window=window_function,
                                    center=True,
                                    dtype=np.complex64,
                                    pad_mode='reflect')

    if expected_mel_shape_2 in stft_matrix.shape and expected_dimensions in \
            stft_matrix.shape:
        print(f"Pass: Expected STFT Matrix dimensions "
              f"{expected_mel_shape_2, expected_dimensions}, Received: "
              f"{stft_matrix.shape}")
    else:
        print('Error STFT matrix dimensions not equal')
        return False

    spec_test = np.dot(mel_filter, np.abs(stft_matrix) ** 2)
    log_spec_test = librosa.core.power_to_db(spec_test, ref=1.0, amin=1e-10,
                                             top_db=None)
    log_spec_test = log_spec_test.astype(np.float32)
    afe = audio_feature_extractor.LogMelExtractor(sample_rate=sample_rate,
                                                  window_size=window_size,
                                                  hop_size=hop_size,
                                                  mel_bins=mel_bins,
                                                  fmin=f_min, fmax=f_max)
    logmel_spec = afe.transform(audio_file)
    if np.allclose(log_spec_test, logmel_spec, 1e-05):
        return True
    else:
        print('Test logmel spectrograms are not similar enough')
        return False


def create_delta_test(test, gt):
    returned = audio_feature_extractor.create_delta(test)
    returned = returned.astype(np.float16)
    gt = gt.astype(np.float16)

    for i, d in enumerate(returned):
        for j, s in enumerate(d):
            for k, a in enumerate(s):
                if a != gt[i, j, k]:
                    return False
    return True


def run_tests_afe(config):
    start = time.time()
    window_sizes = [1024, 1024, 512, 1024]
    hop_sizes = [512, 676, 342, 1024]
    for i in range(len(window_sizes)):
        passed = logmel_transform(window_sizes[i], hop_sizes[i], config)
        if not passed:
            break
    end = time.time()
    if passed:
        print('Test logmel_transform Passed in: ', (end-start), 's')
    else:
        print('Test logmel_transform Failed in: ', (end-start), 's')

    start = time.time()
    dimensions = [1028, 1028, 512, 512]
    overlap = [0, 50, 0, 50]
    for i in range(len(dimensions)):
        passed = segmenter_test(dimensions[i], overlap[i], config)
    end = time.time()
    if passed:
        print('Test segmenter_test Passed in: ', (end-start), 's')
    else:
        print('Test segmenter_test Failed in: ', (end-start), 's')

    start = time.time()
    test = np.array([[5, 5, 5, 6, 6], [6, 7, 8, 8, 8], [9, 9, 8, 9, 8]])
    gt = np.array([[[5, 5, 5, 6, 6], [6, 7, 8, 8, 8], [9, 9, 8, 9, 8]],
                   [[1.5, 1.2, .3, -.9, -1.6], [2.3, 1.8, .5, -1.4, -2.4],
                    [2.5, 1.7, -.2, -1.8, -2.5]],
                   [[.18, -.3, -.83, -.43, .03], [.28, -.46, -1.26, -.65,
                                                  .04], [.13, -.63, -1.35,
                                                         -.57, .22]]])
    passed = create_delta_test(test, gt)
    end = time.time()
    if passed:
        print('Test segmenter_test Passed in: ', (end-start), 's')
    else:
        print('Test segmenter_test Failed in: ', (end-start), 's')
