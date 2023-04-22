import main 
import torch
import os

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_float32_matmul_precision('high')

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


    lr = [5e-5]
    batch_sizes = [5]
    gpus = [1]
    # emotion_encoder_name_list = ['j-hartmann/emotion-english-roberta-large'] , j-hartmann/emotion-english-distilroberta-base
    # cause_encoder_name_list = ['roberta-base']
    
    encoder_name = 'j-hartmann/emotion-english-distilroberta-base'
    mode = 'train'
    
    if mode == 'train':
        for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
            for lr_ in lr:
                for batch_size in batch_sizes:
                    runner = main.Main()
                    runner.set_dataset(tr, va, te, dl)
                    runner.set_gpus(gpus)
                    runner.set_hyperparameters(learning_rate=lr_, batch_size=batch_size)
                    runner.set_value('training_iter', 15)
                    runner.set_value('encoder_name', encoder_name)
                    runner.set_value('accumulate_grad_batches', 1)
                    
                    encoder_filename = encoder_name.split('/')[-1]
                    runner.set_value('log_folder_name', f'Encoder-{encoder_filename}_Total_Test_{dl}_batch{batch_size}')
                    runner.run()
                    
                    del runner
    else: # test
        test_model_list = [
            # separated 0,1,2,3,4 / unseparated 5,6,7,8,9
            'model/bert-base-cased--original_data_DailyDialog-not_separated_lr_5e-05_2023-04-16 07:20:14.826287.ckpt',
            'model/bert-base-cased--data_1_DailyDialog-not_separated_lr_5e-05_2023-04-16 07:52:05.793635.ckpt',
            'model/bert-base-cased--data_2_DailyDialog-not_separated_lr_5e-05_2023-04-16 08:26:40.517597.ckpt',
            'model/bert-base-cased--data_3_DailyDialog-not_separated_lr_5e-05_2023-04-16 09:00:43.799727.ckpt',
            'model/bert-base-cased--data_4_DailyDialog-not_separated_lr_5e-05_2023-04-16 09:36:27.538797.ckpt',
        ]
        test_log_folder_list = [
            'test_bert-unseparated_at_original',
            'test_bert-unseparated_at_data1',
            'test_bert-unseparated_at_data2',
            'test_bert-unseparated_at_data3',
            'test_bert-unseparated_at_data4',
        ]
        for tr, va, te, dl, tm, tfn in zip(train_data_list, valid_data_list, test_data_list, 
                                      data_label, test_model_list, test_log_folder_list):
            runner = main.Main()
            emotion_encoder_name = emotion_encoder_name_list[0]
            cause_encoder_name = cause_encoder_name_list[0]
            runner.set_dataset(tr, va, te, dl)
            runner.set_gpus(gpus)
            runner.set_value('emotion_encoder_name', emotion_encoder_name)
            runner.set_value('cause_encoder_name', cause_encoder_name)
            runner.set_test(ckpt_path=tm)
            runner.set_value('encoder_separation', False)
            runner.set_value('log_folder_name', tfn)
            runner.run()
            
            del runner
        