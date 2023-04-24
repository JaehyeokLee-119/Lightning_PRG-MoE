import argparse
import os
import random
from typing import List

import numpy as np
import torch

from datetime import datetime

from module.trainer import LearningEnv
from dotenv import load_dotenv

def set_random_seed(seed: int):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='This code is for ECPE task.')

    # Training Environment
    parser.add_argument('--gpus', default="1")
    parser.add_argument('--num_process', default=int(os.cpu_count() * 0.8), type=int)
    parser.add_argument('--num_worker', default=6, type=int)
    parser.add_argument('--port', default=1234, type=int)

    parser.add_argument('--model_name', default='PRG_MoE')
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--test', default=False)

    parser.add_argument('--train_data', default="data/data_fold/data_0/dailydialog_train.json")
    parser.add_argument('--valid_data', default="data/data_fold/data_0/dailydialog_valid.json")
    parser.add_argument('--test_data', default="data/data_fold/data_0/dailydialog_test.json")
    parser.add_argument('--log_directory', default='logs', type=str)
    parser.add_argument('--log_folder_name', default=None, type=str)
    parser.add_argument('--data_label', help='the label that attaches to saved model', default='dailydialog_fold_0')

    parser.add_argument('--only_emotion', help='only emotion classification', default=False, type=bool)
    parser.add_argument('--only_cause', help='only cause classification', default=False, type=bool)
    parser.add_argument('--emo_model_path', help='the path of emotion model', default=None)
    
    # Encoder Model Setting
    parser.add_argument('--encoder_name', help='the name of encoder', default='roberta-base')
    parser.add_argument('--unfreeze', help='the number of layers to be unfrozen', default=0, type=int) # 0 == unfreeze all layers
    parser.add_argument('--max_seq_len', help='the max length of each tokenized utterance', default=75, type=int)
    parser.add_argument('--contain_context', help='While tokenizing, previous utterances are contained or not', default=False)
    parser.add_argument('--loss_lambda', help='Ratio of emotion loss in the total loss', default=0.2)
    
    # 원래의 PRG-MoE 모델을 사용한다 (데이터 비교를 위해)
    parser.add_argument('--use_original', default=False)
    
    # Cause 부분에 FC Layer 추가를 사용한다
    parser.add_argument('--use_newfc', default=False, type=bool)

    # Training Hyperparameters
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--guiding_lambda', help='the mixing ratio', default=0.6, type=float)
    parser.add_argument('--training_iter', default=8, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--patience', help='patience for Early Stopping', default=None, type=int)
    parser.add_argument('--encoder_separation', help='Use encoder separation or not', default=False, type=bool)
    parser.add_argument('--emotion_epoch_ratio', default=0.5, type=float)
    parser.add_argument('--accumulate_grad_batches', default=2, type=int)
    
    parser.add_argument('--n_speaker', help='the number of speakers', default=2, type=int)
    parser.add_argument('--n_emotion', help='the number of emotions', default=7, type=int)
    parser.add_argument('--n_cause', help='the number of causes', default=2, type=int)
    parser.add_argument('--n_expert', help='the number of causes', default=4, type=int)

    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--wandb_project_name', default='ECPE')
    return parser.parse_args()


def test_preconditions(args: argparse.Namespace):
    if args.test:
        assert args.pretrained_model is not None, "For test, you should load pretrained model."


class Main:
    def __init__(self):
        load_dotenv()
        self.args = parse_args()
        test_preconditions(self.args)
        set_random_seed(77)
        
    def run(self):
        # Start Training/Testing
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in self.args.gpus])
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore TensorFlow error message 
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ["WANDB_DISABLED"] = "true"
        
        if type(self.args.gpus) == str: # gpu가 문자열이면 list로 파싱
            self.args.gpus = self.args.gpus.split(',')
            self.args.gpus = [int(_) for _ in self.args.gpus]
            
        encoder_name = self.args.encoder_name.replace('/', '_')  
        self.args.wandb_pjname = f'ECPE_{encoder_name}_lr{self.args.learning_rate}_Unfreze{self.args.unfreeze}_{self.args.data_label}'
        
        # 실행
        trainer = LearningEnv(**vars(self.args))
        trainer.run(**vars(self.args))
        del trainer
        
        torch.cuda.empty_cache()
    
    def set_dataset(self, train_dataset, valid_dataset, test_dataset, data_label):
        self.args.train_data = train_dataset
        self.args.valid_data = valid_dataset
        self.args.test_data = test_dataset
        self.args.data_label = data_label
        
    def set_gpus(self, gpus):
        self.args.gpus = gpus
        
    def set_test(self, ckpt_path):
        self.args.test = True
        self.args.pretrained_model = ckpt_path
    
    def set_hyperparameters(self, learning_rate, batch_size):
        self.args.learning_rate = learning_rate
        self.args.batch_size = batch_size
    
    def set_value(self, key, value):
        setattr(self.args, key, value)
    
    def get_value(self, key):
        return getattr(self.args, key)
    
if __name__ == "__main__":
    main = Main()
    main.run()
    
