import logging
import os
import datetime
import tensorflow
import lightning.pytorch as L
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist

from module.lightmodule import LitPRGMoE

from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import module.model as M
from module.evaluation import log_metrics, FocalLoss
from module.preprocessing import get_data, tokenize_conversation, get_pad_idx, get_pair_pad_idx
from module.custom_trainer import CustomTrainer, CustomDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore TF error message 

class LearningEnv:
    def __init__(self, **kwargs):
        self.use_wandb = kwargs['use_wandb']
        self.wandb_project_name = kwargs['wandb_pjname']
        self.num_worker = kwargs['num_worker']
        # if self.use_wandb:
        #     wandb.init(project=self.wandb_project_name)
        self.only_emotion = kwargs['only_emotion']
        
        self.gpus = kwargs['gpus']
        self.single_gpu = len(self.gpus) == 1
        self.num_worker = kwargs['num_worker']

        self.train_dataset = kwargs['train_data']
        self.valid_dataset = kwargs['valid_data']
        self.test_dataset = kwargs['test_data']
        self.data_label = kwargs['data_label']

        self.max_seq_len = kwargs['max_seq_len']
        self.start_time = datetime.datetime.now()
        
        self.log_directory = kwargs['log_directory']

        self.training_iter = kwargs['training_iter']
        
        self.model_name = kwargs['model_name']
        self.port = kwargs['port']
        
        self.pretrained_model = kwargs['pretrained_model']
        
        # Hyperparameters
        self.dropout = kwargs['dropout']
        self.n_cause = kwargs['n_cause']
        self.n_speaker = kwargs['n_speaker']
        self.n_emotion = kwargs['n_emotion']
        self.n_expert = kwargs['n_expert']
        self.encoder_name = kwargs['encoder_name']
        self.learning_rate = kwargs['learning_rate']
        self.unfreeze = kwargs['unfreeze']
        self.batch_size = kwargs['batch_size']
        self.guiding_lambda = kwargs['guiding_lambda']
        self.contain_context = kwargs['contain_context']
        
        # learning variables
        self.best_performance = [0, 0, 0]  # p, r, f1
        self.num_epoch = 1
        
        self.model_args = {
            "dropout": self.dropout,
            "n_speaker": self.n_speaker,
            "n_emotion": self.n_emotion,
            "n_cause": self.n_cause,
            "n_expert": self.n_expert,
            "guiding_lambda": self.guiding_lambda,
            "learning_rate": self.learning_rate,
            "encoder_name": self.encoder_name,
            "unfreeze": self.unfreeze,
            "only_emotion": self.only_emotion,
            "training_iter": self.training_iter,
        }

    def __set_model__(self):
        model_args = self.model_args

        if self.pretrained_model is not None:
            model = getattr(M, self.model_name)(**model_args)
            model.load_state_dict(torch.load(self.pretrained_model))
        else:
            model = getattr(M, self.model_name)(**model_args)

        return model

    def set_model(self):
        self.model = self.__set_model__()
        
    def set_logger_environment(self, file_name_list, logger_name_list):
        for file_name, logger_name in zip(file_name_list, logger_name_list):
            for handler in logging.getLogger(logger_name).handlers[:]:
                logging.getLogger(logger_name).removeHandler(handler)
            self.set_logger(file_name, logger_name)

    def set_logger(self, file_name, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if self.log_directory:
            if not os.path.exists(f'{self.log_directory}'):
                os.makedirs(f'{self.log_directory}')
            file_handler = logging.FileHandler(f'{self.log_directory}/{file_name}')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def get_dataloader(self, dataset_file, batch_size, num_worker, shuffle=True, contain_context=False):
        device = "cuda:0"
        data = get_data(dataset_file, device, self.max_seq_len, self.encoder_name, contain_context)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = data[0]
        speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = data[1:]

        dataset_ = TensorDataset(utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
        dataloader_params = {
            "dataset": dataset_,
            "batch_size": batch_size,
            # "shuffle": shuffle
        }
        
        return DataLoader(**dataloader_params)
    
    def get_dataset(self, dataset_file):
        device = "cuda:0"
        data = get_data(dataset_file, device, self.max_seq_len, self.encoder_name, self.contain_context)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = data[0]
        speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = data[1:]

        dataset_ = TensorDataset(utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
        return dataset_
    
    def run(self, **kwargs):
        self.pre_setting()
        if kwargs['test']:
            self.training_iter = 1
            self.valid(type='test')
        else:
            self.train()
    
    def pre_setting(self):
        # 로그 폴더 생성 및 로거 설정
        logger_name_list = ['train', 'valid', 'test']
        encoder_name = self.encoder_name.replace('/', '_')
        
        EmotionText = 'OnlyEmotion' if self.only_emotion else ''
        file_name_list = [f'{encoder_name}-{EmotionText}{self.data_label}-lr_{self.learning_rate}-Unfreeze{self.unfreeze}-{_}-{self.start_time}.log' for _ in logger_name_list]
        self.set_logger_environment(file_name_list, logger_name_list)
        
        # 모델 저장할 폴더 생성
        if not os.path.exists("model/"):
            os.makedirs("model/")
            
        # 모델 저장기를 설정
        self.saver = ModelSaver(path=f"model/{encoder_name}-{self.data_label}-lr_{self.learning_rate}-Unfreeze{self.unfreeze}.pt", single_gpu=self.single_gpu)
        
        # 모델 인스턴스를 셋팅
        self.set_model()
    
        
    def train(self, train_type='cause'):
        logger = logging.getLogger('train')
        
        train_dataset_custom = CustomDataset(self.train_dataset, max_seq_len=self.max_seq_len, encoder_name=self.encoder_name, contain_context=self.contain_context)
        valid_dataset_custom = CustomDataset(self.valid_dataset, max_seq_len=self.max_seq_len, encoder_name=self.encoder_name, contain_context=self.contain_context)
        test_dataset_custom = CustomDataset(self.test_dataset, max_seq_len=self.max_seq_len, encoder_name=self.encoder_name, contain_context=self.contain_context)
        
        train_dataloader = DataLoader(train_dataset_custom, batch_size=self.batch_size, num_workers=self.num_worker)
        valid_dataloader = DataLoader(valid_dataset_custom, batch_size=64, num_workers=self.num_worker)
        test_dataloader = DataLoader(test_dataset_custom, batch_size=64, num_workers=self.num_worker)
                    
        model = LitPRGMoE(**self.model_args)
        trainer = L.Trainer(max_epochs=8)
        trainer.fit(model, train_dataloaders=train_dataloader)
    
    def valid(self, type='cause', option='valid'):
        if self.only_emotion:
            type = 'emotion'
        else:
            type = 'cause'
            
        if option == 'valid':
            dataset = self.valid_dataset
        else:
            dataset = self.test_dataset

        logger = logging.getLogger(option)
        
        with torch.no_grad():
            valid_dataloader = self.get_dataloader(dataset, self.batch_size, self.num_worker, shuffle=False, contain_context=self.contain_context)
            self.model.eval()
            loss_avg, count= 0, 0
            emo_pred_y_list, emo_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, cau_pred_y_list, cau_true_y_list = [list() for _ in range(6)]

            for utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch in tqdm(valid_dataloader, desc=f"{option}"):
                batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
                
                # Get Model output  
                prediction = self.model(
                                                    utterance_input_ids_batch, 
                                                    utterance_attention_mask_batch, 
                                                    utterance_token_type_ids_batch, 
                                                    speaker_batch
                                                    )
                emotion_prediction, pair_binary_cause_prediction = prediction
                
                # Output processing
                check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.encoder_name)
                check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=3, emotion_pred=emotion_prediction)
                check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=1000,)
                
                # Emotion Prediction, label
                emotion_prediction = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
                emotion_label_batch = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]
                
                # Cause Prediction, label
                pair_binary_cause_prediction_window = pair_binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                pair_binary_cause_prediction_all = pair_binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
                
                pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
                pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
                
                device = torch.device("cuda")
                
                # Loss Calculation
                if type == 'cause':
                    criterion_emo = FocalLoss(gamma=2)
                    criterion_cau = FocalLoss(gamma=2)
                    
                    loss_emo = criterion_emo(emotion_prediction, emotion_label_batch)
                    if (torch.sum(check_pair_window_idx)==0):
                        loss_cau = torch.tensor(0.0)
                    else:
                        loss_cau = criterion_cau(pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
                    loss = 0.2 * loss_emo + 0.8 * loss_cau
                elif type == 'emotion':
                    criterion_emo = FocalLoss(gamma=2)
                    loss_emo = criterion_emo(emotion_prediction, emotion_label_batch)
                    loss = loss_emo
                    
                # Logging
                cau_pred_y_list_all.append(pair_binary_cause_prediction_all), cau_true_y_list_all.append(pair_binary_cause_label_batch_all)
                cau_pred_y_list.append(pair_binary_cause_prediction_window), cau_true_y_list.append(pair_binary_cause_label_batch_window)
                emo_pred_y_list.append(emotion_prediction), emo_true_y_list.append(emotion_label_batch)

                loss_avg += loss.item()
                count += 1
                
            loss_avg = loss_avg / count
            
            # Log Metrics
            logger.info(f'\nEpoch: [{self.num_epoch}/{self.training_iter}]')
            p_cau, r_cau, f1_cau = log_metrics(logger, emo_pred_y_list, emo_true_y_list, cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, loss_avg, n_cause=self.n_cause, option=option)
            
            del valid_dataloader
            
            # 'Valid'인 경우, best performance인지 확인 후 모델 저장
            if option == 'valid':
                if type == 'cause':
                    # 감정-원인 분류인 경우
                    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                elif type == 'emotion':
                    # 감정 분류인 경우
                    pass
                if self.best_performance[-1] < f1_cau:
                    self.saver(self.model) # save model when best performance
            
            # 'Test'인 경우, best performance를 저장하고 로깅
            if option == 'test':
                if type == 'cause':
                    # 감정-원인 분류인 경우
                    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
                elif type == 'emotion':
                    # 감정 분류인 경우
                    pass
                    
                if self.best_performance[-1] < f1_cau:
                    self.best_performance = [p_cau, r_cau, f1_cau]
                    
                p, r, f1 = self.best_performance
                
                if type == 'cause':
                    logger.info(f'\n[current best performance] precision: {p} | recall: {r} | f1-score: {f1}\n') # For Cause + Emotion
                else:
                    logger.info(f'\n[current best emotion performance (f1)] accuracy: {p} | macro avg: {r} | weighted avg: {f1}\n') # For Only Emotion
            
    

class ModelSaver:
    def __init__(self, path='checkpoint.pt', single_gpu=None):
        self.path = path
        self.single_gpu = single_gpu

    def __call__(self, model):
        state_dict = model.state_dict()
        torch.save(state_dict, self.path)