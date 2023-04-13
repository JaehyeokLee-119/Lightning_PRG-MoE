import main 
import torch
import os

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_float32_matmul_precision('high')

    # Original Dataset (1 fold)
    train_data_list = ['data/data_fold/data_0/dailydialog_train.json']
    valid_data_list = ['data/data_fold/data_0/dailydialog_valid.json']
    test_data_list = ['data/data_fold/data_0/dailydialog_test.json']
    data_label = ['-original_fold']
    
    # # Mini Dataset (1 fold)
    # train_data_list = ['data/data_mini/dailydialog_train.json']
    # valid_data_list = ['data/data_mini/dailydialog_valid.json']
    # test_data_list = ['data/data_mini/dailydialog_test.json']
    # data_label = ['-original_mini']
    
    gpus = [1]
    encoder_name = 'roberta-large'
    for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
        runner = main.Main()
        runner.set_dataset(tr, va, te, dl)
        runner.set_gpus(gpus)
        runner.set_encoder_name(encoder_name)
        # runner.set_test(ckpt_path='lightning_logs/version_0/checkpoints/epoch=7-step=3336.ckpt')
        runner.run()
        
        del runner