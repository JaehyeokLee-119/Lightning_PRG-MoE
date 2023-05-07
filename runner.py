import main 
import torch
import os
import datetime

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
    
    # # Original Dataset (1 fold)
    # train_data_list = ['data/data_fold/data_0/dailydialog_train.json']
    # valid_data_list = ['data/data_fold/data_0/dailydialog_valid.json']
    # test_data_list = ['data/data_fold/data_0/dailydialog_test.json']
    # data_label = ['-original_fold']
    
    start_time = datetime.datetime.now()
    
    lr = [5e-5]
    batch_sizes = [5]
    gpus = [1]
    loss_lambda = 0.2
    accumulate_grad_batches = 1
    # emotion_encoder_name_list = ['j-hartmann/emotion-english-roberta-large'] , j-hartmann/emotion-english-distilroberta-base
    # cause_encoder_name_list = ['roberta-base']
    
        # encoder_name이 ORIGINAL이면, Original PRG-MoE(BertModel)를 사용하고, 아니면, 
        # 해당 이름의 모델(AutoModelForSequenceClassification)을 사용한다.
    encoder_name_list = ['bert-base-cased'] #['distilroberta-base',]# ['bert-base-cased']#
    encoder_label_list = ['PRG-MoE(BERT)-Contexted'] #['Distilroberta-base', ] #['PRG-MoE(BERT)']#
    mode = 'train'
    use_newfc = False
    epoch = 20
    ckpt_type = 'cause-f1' # 'cause-f1', 'emotion-f1', 'joint-f1'
    model_save_path = "/hdd/hjl8708/0507-context"
    multiclass_avg_type = 'micro'
    window_size = 3
    freeze_ratio = 0
    
    contain_context = True
    max_seq_len = 75
    
    if mode == 'train':
        for encoder_name, encoder_label in zip(encoder_name_list, encoder_label_list):
            for tr, va, te, dl in zip(train_data_list, valid_data_list, test_data_list, data_label):
                for lr_ in lr:
                    for batch_size in batch_sizes:
                        runner = main.Main()
                        runner.set_dataset(tr, va, te, dl)
                        runner.set_gpus(gpus)
                        runner.set_hyperparameters(learning_rate=lr_, batch_size=batch_size)
                        
                        runner.set_value('contain_context', contain_context)
                        runner.set_value('max_seq_len', max_seq_len)
                        
                        runner.set_value('training_iter', epoch)
                        runner.set_value('encoder_name', encoder_name)
                        runner.set_value('model_save_path', model_save_path)
                        runner.set_value('accumulate_grad_batches', accumulate_grad_batches)
                        runner.set_value('loss_lambda', loss_lambda)
                        runner.set_value('ckpt_type', ckpt_type)
                        runner.set_value('freeze_ratio', freeze_ratio)
                        runner.set_value('use_newfc', use_newfc)
                        runner.set_value('multiclass_avg_type', multiclass_avg_type)
                        runner.set_value('log_directory', 'logs')
                        runner.set_value('window_size', window_size)
                        encoder_name_for_filename = encoder_name.replace('/', '-')
                        # runner.set_value('log_folder_name', f'Encoder_loss_lambda{loss_lambda}-{encoder_filename}_Total_Test_{dl}_batch{batch_size}')
                        runner.set_value('log_folder_name', f'Gpu{gpus}{encoder_label} Epoch{epoch}: for BEST {multiclass_avg_type} {ckpt_type},losslambda{loss_lambda}, UseNewFC-{use_newfc}_{dl}-{start_time}')
                        runner.run()
                        
                        del runner
    else: # test
        test_model_list = [
            # separated 0,1,2,3,4 / unseparated 5,6,7,8,9
            * [f'/hdd/hjl8708/0424-lightning/(모델타입:BEST joint_accuracy, UseNewFC-True-batch5,축적1-losslambda0.2)bert-base-cased_-data_{fold_}_DailyDialog.ckpt' for fold_ in range(1, 5)],
            '/hdd/hjl8708/0424-lightning/(모델타입:BEST joint_accuracy, UseNewFC-True-batch5,축적1-losslambda0.2)bert-base-cased_-original_data_DailyDialog.ckpt'
        ]
        test_log_folder_list = [
            * [f'TEST_bert-joint_accuracy-at_data{fold_}' for fold_ in range(1, 5)],
            'TEST_bert-joint_accuracy-at_original',
        ]
        encoder_name = 'bert-base-cased'
        for tr, va, te, dl, tm, tfn in zip(train_data_list, valid_data_list, test_data_list, 
                                      data_label, test_model_list, test_log_folder_list):
            runner = main.Main()
            runner.set_dataset(tr, va, te, dl)
            runner.set_gpus(gpus)
            runner.set_value('encoder_name', encoder_name)
            runner.set_test(ckpt_path=tm)
            runner.set_value('use_newfc', True)
            runner.set_value('log_folder_name', tfn)
            runner.run()
            
            del runner
        