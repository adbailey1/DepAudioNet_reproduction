SEPARATOR = '###############################################################'

label = {'Not_Depressed', 'Depressed'}
LABELS = {label: i for i, label in enumerate(label)}
INDEX = {i: label for i, label in enumerate(label)}

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
                'val_false_negative', 'val_true_positive']
