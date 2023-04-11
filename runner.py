import main 
import torch
import os

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Original Dataset (1 fold)
    train_data_list = ['data/data_fold/data_0/dailydialog_train.json']
    valid_data_list = ['data/data_fold/data_0/dailydialog_valid.json']
    test_data_list = ['data/data_fold/data_0/dailydialog_test.json']
    data_label = ['-original_fold']
    
    # Mini Dataset (1 fold)
    train_data_list = ['data/data_mini/dailydialog_train.json']
    valid_data_list = ['data/data_mini/dailydialog_valid.json']
    test_data_list = ['data/data_mini/dailydialog_test.json']
    data_label = ['-original_mini']
    
    torch.set_float32_matmul_precision('high')
    
    for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
        runner = main.Main()
        runner.set_dataset(tr, va, te, dl)
        runner.run()
        
        del runner