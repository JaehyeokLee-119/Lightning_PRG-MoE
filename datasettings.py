# Entire data
train_data_list = [
    'data/data_fold/data_0/dailydialog_train.json',
    * [f'data/data_fold/data_{fold_}/data_{fold_}_train.json' for fold_ in range(1, 5)]
]
valid_data_list = [
    'data/data_fold/data_0/dailydialog_valid.json',
    * [f'data/data_fold/data_{fold_}/data_{fold_}_valid.json' for fold_ in range(1, 5)]
]
test_data_list = [
    'data/data_fold/data_0/dailydialog_test.json',
    * [f'data/data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]
]
data_label = ['-original_data_DailyDialog', *[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)]]

# Entire data (folds 먼저)
train_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_train.json' for fold_ in range(1, 5)],
    'data/data_fold/data_0/dailydialog_train.json',
]
valid_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_valid.json' for fold_ in range(1, 5)],
    'data/data_fold/data_0/dailydialog_valid.json',
]
test_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)],
    'data/data_fold/data_0/dailydialog_test.json',
]
data_label = [*[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)], '-original_data_DailyDialog']

    
# Original Dataset (1 fold)
train_data_list = ['data/data_fold/data_0/dailydialog_train.json']
valid_data_list = ['data/data_fold/data_0/dailydialog_valid.json']
test_data_list = ['data/data_fold/data_0/dailydialog_test.json']
data_label = ['-original_fold']

# Another folds
fold_ = 1
train_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_train.json']
]
valid_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_valid.json']
]
test_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_test.json']
]
data_label = [* [f'-data_{fold_}_DailyDialog']]

# Folds
train_data_list = [
* [f'data/data_fold/data_{fold_}/data_{fold_}_train.json' for fold_ in range(1, 5)]
]
valid_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_valid.json' for fold_ in range(1, 5)]
]
test_data_list = [
    * [f'data/data_fold/data_{fold_}/data_{fold_}_test.json' for fold_ in range(1, 5)]
]
data_label = [*[f'-data_{fold_}_DailyDialog' for fold_ in range(1, 5)]]

# Mini Dataset (1 fold)
train_data_list = ['data/data_mini/dailydialog_train.json']
valid_data_list = ['data/data_mini/dailydialog_valid.json']
test_data_list = ['data/data_mini/dailydialog_test.json']
data_label = ['-original_mini']

# ConvECPE
train_data_list = ['ConvECPE/ConvECPE_train.json',]
valid_data_list = ['ConvECPE/ConvECPE_valid.json',]
test_data_list = ['ConvECPE/ConvECPE_test.json',]
data_label = ['-ConvECPE',]

# ConvECPE 5-fold cross-validation
train_data_list = [
* [f'data/data_ConvECPE/ConvECPE_fold_{fold_}_train.json' for fold_ in range(0, 5)]
]
valid_data_list = [
    * [f'data/data_ConvECPE/ConvECPE_fold_{fold_}_valid.json' for fold_ in range(0, 5)]
]
test_data_list = [
    * [f'data/data_ConvECPE/ConvECPE_fold_{fold_}_test.json' for fold_ in range(0, 5)]
]
data_label = [*[f'-data_{fold_}_ConvECPE' for fold_ in range(0, 5)]]