import main 
import torch
import os

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_float32_matmul_precision('high')

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

        
    lr = 5e-5
    batch_size = 4
    gpus = [1]
    encoder_name = 'bert-base-cased'
    sep_encoder_list = [True, False]
    
    for sep_encoder in sep_encoder_list:
        for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
            runner = main.Main()
            runner.set_dataset(tr, va, te, dl)
            runner.set_gpus(gpus)
            runner.set_encoder_name(encoder_name)
            runner.set_hyperparameters(learning_rate=lr, batch_size=batch_size)
            # runner.set_test(ckpt_path='lightning_logs/version_0/checkpoints/epoch=7-step=3336.ckpt')
            runner.set_value('encoder_separation', sep_encoder)
            runner.run()
            
            del runner