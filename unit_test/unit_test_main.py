import numpy as np
import sklearn.metrics as metrics
import main


def prediction_and_accuracy_test(config):
    batch_output = np.array(([0,1],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[1,0],
                            [0,1],[1,0]))
    batch_out_arg = np.array(([1], [0], [0], [0], [0], [0], [1], [0], [1],
                              [0]))
    batch_labels = np.array(([1], [1], [1], [1], [0], [1], [0], [0], [0], [0]))
    initial_condition = 1
    number_classes = 2
    comp_res = np.zeros(30)
    loss = per_epoch_pred = 0
    gt_per_epoch_pred = np.hstack((batch_output, batch_labels))
    gt_acc = np.array([0.6, 0.2])
    gt_fscore = metrics.precision_recall_fscore_support(batch_labels,
                                                        batch_out_arg)
    gt_fscore = np.array(gt_fscore[0:3]).reshape(1, -1)[0]
    gt_tn_fp_fn_tp = np.array([3, 2, 4, 1])

    comp_res, per_epoch_pred = main.prediction_and_accuracy(batch_output, batch_labels,
                                                            initial_condition,
                                                            number_classes, comp_res, loss,
                                                            per_epoch_pred, config)

    if np.all((comp_res[0:2] == gt_acc)) and np.all((comp_res[2:8] ==
                                                     gt_fscore)) and \
            np.all((comp_res[11:15] == gt_tn_fp_fn_tp)):
        error = 0
        for i, d in enumerate(per_epoch_pred):
            for j, c in enumerate(d):
                if c != gt_per_epoch_pred[i, j]:
                    error += 1
        if error == 0:
            print('Test Passed')
            print('Expected Accuracy: ', gt_acc, 'Received Accuracy: ', comp_res[
                                                                    0:2])
            print('Expected FScore: ', gt_fscore, 'Received FScore: ', comp_res[
                                                                  2:8])
            print('Expected Confusion Matrix: ', gt_tn_fp_fn_tp, 'Received '
                                                             'Confusion '
                                                             'Matrix',
              comp_res[11:15])
            return True
        else:
            print('Test Failed')
            print('Expected Accuracy: ', gt_acc, 'Received Accuracy: ', comp_res[
                                                                        0:2])
            print('Expected FScore: ', gt_fscore, 'Received FScore: ', comp_res[
                                                                      2:8])
            print('Expected Confusion Matrix: ', gt_tn_fp_fn_tp, 'Received '
                                                                 'Confusion '
                                                                 'Matrix',
                  comp_res[11:15])
            return False


def run_tests_m(config):
    result = prediction_and_accuracy_test(config)
