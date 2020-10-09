from unit_test.unit_test_audio_feature_extractor import run_tests_afe
from unit_test.unit_test_dataset_processing import run_tests_dp
from unit_test.unit_test_data_gen import run_tests_dg
from unit_test.unit_test_organiser import run_tests_o
from unit_test.unit_test_main import run_tests_m
import os
import numpy as np
import random
import time
from unit_test import config_test


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    np.random.seed(1234)
    random.seed(1234)
    total_time_start = time.time()
    # run_tests_afe(config_test)
    # config_path = 'config'
    # run_tests_dp(config_test, current_path, config_path)
    # run_tests_dg(config_path, current_path)
    # run_tests_o(config_test)
    run_tests_m(config_test)
    total_time_end = time.time()
    print('Unit Tests Took: ', total_time_end-total_time_start, 's')


if __name__ == "__main__":
    main()
