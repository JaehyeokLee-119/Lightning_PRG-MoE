import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup
from module.evaluation import FocalLoss #log_metrics, 
from sklearn.metrics import classification_report, precision_score , recall_score , confusion_matrix
from transformers import AutoModel
import numpy as np
from module.lightmodels import TotalModel, OriginalPRG_MoE, TotalModel_exp12

class LitPRGMoE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # 모델 셋팅 파라미터
        self.encoder_name = kwargs['encoder_name']
        # Model
        
        self.use_original = kwargs['use_original']
        self.use_exp12 = kwargs['use_exp12']
        
        self.freeze_ratio = kwargs['freeze_ratio']
        if self.use_original:
            self.model = OriginalPRG_MoE() # output: (emotion prediction, cause prediction)
        else:
            if self.use_exp12:
                self.model = TotalModel_exp12(self.encoder_name, freeze_ratio=self.freeze_ratio) # output: (emotion prediction, cause prediction)
            else:
                self.model = TotalModel(self.encoder_name, freeze_ratio=self.freeze_ratio) # output: (emotion prediction, cause prediction)

        # 하이퍼파라미터 설정
        self.training_iter = kwargs['training_iter']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.window_size = kwargs['window_size']
        self.n_expert = 4
        self.n_emotion = 7
        self.guiding_lambda = kwargs['guiding_lambda']
        self.loss_lambda = kwargs['loss_lambda'] # loss 중 Emotion loss의 비율
        # 학습 방법 설정
        self.n_cause = kwargs['n_cause']
        self.ckpt_type = kwargs['ckpt_type']
        self.multiclass_avg_type = kwargs['multiclass_avg_type']
        
        # 모델 내 학습 중 변수 설정
        self.test = False # True when testing(on_test_epoch_start ~ on_test_epoch_end)
            # test에서 joint_accuracy를 계산
        self.train_type = 'total'
        
        if 'bert-base' in self.encoder_name:
            self.is_bert_like = True
        else:
            self.is_bert_like = False
                        
        # Dictionaries for logging
        types = ['train', 'valid', 'test']
        self.emo_pred_y_list = {}
        self.emo_true_y_list = {}
        self.cau_pred_y_list = {}
        self.cau_true_y_list = {}
        self.cau_pred_y_list_all = {}
        self.cau_true_y_list_all = {}
        self.cau_pred_y_list_all_windowed = {}
        self.emo_cause_pred_y_list = {}
        self.emo_cause_true_y_list = {}
        self.emo_cause_pred_y_list_all = {}
        self.emo_cause_true_y_list_all = {}
        self.loss_sum = {}
        self.batch_count = {}
        
        for i in types:
            self.emo_pred_y_list[i] = []
            self.emo_true_y_list[i] = []
            self.cau_pred_y_list[i] = []
            self.cau_true_y_list[i] = []
            self.cau_pred_y_list_all[i] = []
            self.cau_true_y_list_all[i] = []
            self.cau_pred_y_list_all_windowed[i] = []
            self.emo_cause_pred_y_list[i] = []
            self.emo_cause_true_y_list[i] = []
            self.emo_cause_pred_y_list_all[i] = []
            self.emo_cause_true_y_list_all[i] = []
            self.loss_sum[i] = 0.0
            self.batch_count[i] = 0
        
        self.best_performance_emo = {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'epoch': 0,
            'loss': 0.0,
        }
        self.best_performance_cau = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'epoch': 0,
            'loss': 0.0,
        }
        self.best_performance_emo_cau = {
            'epoch': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
        }
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=5,
                                                    num_training_steps=self.training_iter,
                                                    )
        return [optimizer], [scheduler]
                    
    def forward(self, batch):
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        
        batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
        
        input_ids = utterance_input_ids_batch
        attention_mask = utterance_attention_mask_batch
        token_type_ids = utterance_token_type_ids_batch
        speaker_ids = speaker_batch
        
        # Forward
        emotion_prediction, cause_prediction = self.model(input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len)
        
        return (emotion_prediction, cause_prediction)
    
    def output_processing(self, utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction):
        # 모델의 forward 결과로부터 loss 계산과 로깅을 위한 input 6개를 구해 리턴
        batch_size, _, _ = utterance_input_ids_batch.shape
        # Output processing
        check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=self.window_size, emotion_pred=emotion_prediction)
        check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=1000, )
        check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.encoder_name)

        # emotion pred/label을 pair candidate만큼 늘림
        emotion_list = emotion_prediction.view(batch_size, -1, 7)
        emotion_pair_list = []
        emotion_pred_list = []
        for doc_emotion in emotion_list: # 전체 batch에서 각 doc(대화)을 가져옴
                end_t = 0
                for utt_emotion in doc_emotion: # 각 대화마다 utterance 가져옴
                    emotion_pred_list.append(torch.argmax(utt_emotion))
                    for _ in range(end_t+1): # 
                        emotion_pair_list.append(torch.argmax(utt_emotion)) # 모델의 감정 예측을 index[7->1]화
                    end_t += 1
        # binary_cause_pred_window_full = torch.argmax(binary_cause_prediction.view(batch_size, -1, self.n_cause), dim=-1)
        emotion_label_pair_list = [] 
        for doc_emotion in emotion_label_batch:
            end_t = 0
            for emotion in doc_emotion:
                for _ in range(end_t+1):
                    emotion_label_pair_list.append(emotion)
                end_t += 1
        emotion_pair_pred_expanded = torch.stack(emotion_pair_list).view(batch_size, -1)
        emotion_pair_true_expanded = torch.stack(emotion_label_pair_list).view(batch_size, -1)
        
        
        # Emotion prediction, label
        emotion_prediction_filtered = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
        emotion_label_batch_filtered = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]
        
        # Cause prediction, label
        pair_binary_cause_prediction_window = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        
        # pair_binary_cause_prediction_all_windowed은 binary_cause_prediction에서 check_pair_window_idx가 0인 것은 전부 0으로 바뀐 형태
        pair_prediction_argmax = torch.argmax(binary_cause_prediction, dim=1)
        pair_prediction_argmax_windowed = pair_prediction_argmax.cpu() * check_pair_window_idx.view(-1)
        pair_prediction_argmax_windowed_all = pair_prediction_argmax_windowed.view(batch_size, -1)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_binary_cause_prediction_all = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        # Emotion-cause prediction, label
        pair_emotion_prediction_window = emotion_pair_pred_expanded[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_emotion_prediction_all = emotion_pair_pred_expanded[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        pair_emotion_label_batch_window = emotion_pair_true_expanded[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_emotion_label_batch_all = emotion_pair_true_expanded[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        # Emotion label에 10을 곱함 (0~6 -> 0~60). Cause 여부는 0,1만 있기 때문에 10을 넘길 수 없어서 이래야 충돌 없음
        pair_emotion_cause_prediction_window = pair_emotion_prediction_window*10 + torch.argmax(pair_binary_cause_prediction_window, dim=1)
        pair_emotion_cause_prediction_all = pair_emotion_prediction_all*10 + torch.argmax(pair_binary_cause_prediction_all, dim=1)
        pair_emotion_cause_label_batch_window = pair_emotion_label_batch_window*10 + pair_binary_cause_label_batch_window
        pair_emotion_cause_label_batch_all = pair_emotion_label_batch_all*10 + pair_binary_cause_label_batch_all
        
        emotion_ = (emotion_prediction_filtered, emotion_label_batch_filtered)
        cause_ = (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all)
        emotion_cause_ = (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all)
        return (emotion_, cause_, emotion_cause_)
    
    
    
    def loss_calculation(self, emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window):
        if self.train_type == 'cause':
            # criterion_emo = FocalLoss(gamma=2)
            criterion_cau = FocalLoss(gamma=2)
            
            # loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
            loss_cau = criterion_cau(pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            # loss = 0.2 * loss_emo + 0.8 * loss_cau
            loss = loss_cau
        elif self.train_type == 'emotion':
            criterion_emo = FocalLoss(gamma=2)
            loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
            loss = loss_emo
        elif self.train_type == 'total':
            criterion_emo = FocalLoss(gamma=2)
            criterion_cau = FocalLoss(gamma=2)
            
            loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
            loss_cau = criterion_cau(pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            loss = self.loss_lambda * loss_emo + (1-self.loss_lambda) * loss_cau
        return loss
    
    
    def training_step(self, batch, batch_idx):
        types = 'train'
        # utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        # Output processing
        (emotion_, cause_, emotion_cause_) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        (emotion_prediction_filtered, emotion_label_batch_filtered) = emotion_
        (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all) = cause_
        (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all) = emotion_cause_
        
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.cau_pred_y_list_all_windowed[types].append(pair_prediction_argmax_windowed_all)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        self.emo_cause_pred_y_list[types].append(pair_emotion_cause_prediction_window)
        self.emo_cause_true_y_list[types].append(pair_emotion_cause_label_batch_window)
        self.emo_cause_pred_y_list_all[types].append(pair_emotion_cause_prediction_all)
        self.emo_cause_true_y_list_all[types].append(pair_emotion_cause_label_batch_all)
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        types = 'valid'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        # Output processing
        (emotion_, cause_, emotion_cause_) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        (emotion_prediction_filtered, emotion_label_batch_filtered) = emotion_
        (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all) = cause_
        (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all) = emotion_cause_
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            
        self.log("valid_loss: ", loss, sync_dist=True)
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.cau_pred_y_list_all_windowed[types].append(pair_prediction_argmax_windowed_all)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        self.emo_cause_pred_y_list[types].append(pair_emotion_cause_prediction_window)
        self.emo_cause_true_y_list[types].append(pair_emotion_cause_label_batch_window)
        self.emo_cause_pred_y_list_all[types].append(pair_emotion_cause_prediction_all)
        self.emo_cause_true_y_list_all[types].append(pair_emotion_cause_label_batch_all)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
    def test_step(self, batch, batch_idx):
        types = 'test'
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        # Output processing
        (emotion_, cause_, emotion_cause_) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        (emotion_prediction_filtered, emotion_label_batch_filtered) = emotion_
        (pair_binary_cause_prediction_window, pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all, pair_prediction_argmax_windowed_all) = cause_
        (pair_emotion_cause_prediction_window, pair_emotion_cause_prediction_all, pair_emotion_cause_label_batch_window, pair_emotion_cause_label_batch_all) = emotion_cause_
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            
        self.log("test_loss: ", loss, sync_dist=True)
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.cau_pred_y_list_all_windowed[types].append(pair_prediction_argmax_windowed_all)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        self.emo_cause_pred_y_list[types].append(pair_emotion_cause_prediction_window)
        self.emo_cause_true_y_list[types].append(pair_emotion_cause_label_batch_window)
        self.emo_cause_pred_y_list_all[types].append(pair_emotion_cause_prediction_all)
        self.emo_cause_true_y_list_all[types].append(pair_emotion_cause_label_batch_all)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1

    def on_train_epoch_start(self):
        self.make_test_setting(types='train')
        print('Train type: ', self.train_type)
        
    def on_train_epoch_end(self):
        self.log_test_result(types='train')
    
    def on_validation_epoch_start(self):
        self.test = True
        self.make_test_setting(types='valid')

    def on_validation_epoch_end(self):
        self.test = False
        self.log_test_result(types='valid')
    
    def on_test_epoch_start(self):
        self.test = True
        self.make_test_setting(types='test')
        
    def on_test_epoch_end(self):
        self.test = False
        self.log_test_result(types='test')
        
    def make_test_setting(self, types='train'):
        self.emo_pred_y_list[types] = []
        self.emo_true_y_list[types] = []
        self.cau_pred_y_list[types] = []
        self.cau_true_y_list[types] = []
        self.cau_pred_y_list_all[types] = []
        self.cau_true_y_list_all[types] = []
        self.cau_pred_y_list_all_windowed[types] = []
        self.emo_cause_pred_y_list[types] = []
        self.emo_cause_true_y_list[types] = []
        self.emo_cause_pred_y_list_all[types] = []
        self.emo_cause_true_y_list_all[types] = []
        self.loss_sum[types] = 0.0
        self.batch_count[types] = 0
        
    def log_test_result(self, types='train'):
        logger = logging.getLogger(types)
        
        loss_avg = self.loss_sum[types] / self.batch_count[types]
        emo_report, emo_metrics, emo_binary_report, acc_cau, p_cau, r_cau, f1_cau, p_emo_cau, r_emo_cau, f1_emo_cau = log_metrics(self.emo_pred_y_list[types], self.emo_true_y_list[types], 
                                                self.cau_pred_y_list[types], self.cau_true_y_list[types],
                                                self.cau_pred_y_list_all[types], self.cau_true_y_list_all[types], 
                                                self.cau_pred_y_list_all_windowed[types], 
                                                self.emo_cause_pred_y_list[types], self.emo_cause_true_y_list[types],
                                                self.emo_cause_pred_y_list_all[types], self.emo_cause_true_y_list_all[types],
                                                loss_avg, self.multiclass_avg_type)
        
        self.log('binary_cause 1.loss', loss_avg, sync_dist=True)
        self.log('binary_cause 2.accuracy', acc_cau, sync_dist=True)
        self.log('binary_cause 3.precision', p_cau, sync_dist=True)
        self.log('binary_cause 4.recall', r_cau, sync_dist=True)
        self.log('binary_cause 5.f1-score', f1_cau, sync_dist=True)
        
        self.log('emo 1.accuracy', emo_metrics[0], sync_dist=True)
        self.log('emo 2.macro-f1', emo_metrics[1], sync_dist=True)
        self.log('emo 3.weighted-f1', emo_metrics[2], sync_dist=True)
        
        self.log('emo-cau 1.precision', p_emo_cau, sync_dist=True)
        self.log('emo-cau 2.recall', r_emo_cau, sync_dist=True)
        self.log('emo-cau 3.f1-score', f1_emo_cau, sync_dist=True)
        
        logging_texts = f'\n[Epoch {self.current_epoch}] / <Emotion Prediction> of {types}\n'+\
                        emo_report+\
                        f'\n<Binary-Emotion Prediction Performance>\n'+\
                        emo_binary_report+\
                        f'\n<Emotion-Cause Prediction>'+\
                        f'\n\taccuracy: \t{acc_cau}'+\
                        f'\n\tprecision:\t{p_cau}'+\
                        f'\n\trecall:   \t{r_cau}'+\
                        f'\n\tf1-score: \t{f1_cau}'+\
                        f'\n\n<Pair-Emotion Prediction>'+\
                        f'\n\tprecision:\t{p_emo_cau}'+\
                        f'\n\trecall:   \t{r_emo_cau}'+\
                        f'\n\tf1-score: \t{f1_emo_cau}'+\
                        f'\n'
                        
        if (types == 'valid'):
            if (self.best_performance_emo['weighted_f1'] < emo_metrics[2]):
                self.best_performance_emo['weighted_f1'] = emo_metrics[2]
                self.best_performance_emo['accuracy'] = emo_metrics[0]
                self.best_performance_emo['macro_f1'] = emo_metrics[1]
                self.best_performance_emo['epoch'] = self.current_epoch
                self.best_performance_emo['loss'] = loss_avg
            if (self.best_performance_cau['f1'] < f1_cau):
                self.best_performance_cau['f1'] = f1_cau
                self.best_performance_cau['accuracy'] = acc_cau
                self.best_performance_cau['precision'] = p_cau
                self.best_performance_cau['recall'] = r_cau
                self.best_performance_cau['epoch'] = self.current_epoch
                self.best_performance_cau['loss'] = loss_avg
            if (self.best_performance_emo_cau['f1'] < f1_emo_cau):
                self.best_performance_emo_cau['precision'] = p_emo_cau
                self.best_performance_emo_cau['recall'] = r_emo_cau
                self.best_performance_emo_cau['f1'] = f1_emo_cau
                self.best_performance_emo_cau['epoch'] = self.current_epoch
            
            appended_log_valid = f'\nCurrent Best Performance: loss: {self.best_performance_cau["loss"]}\n'+\
                            f'\t<Emotion Prediction: [Epoch: {self.best_performance_emo["epoch"]}]>\n'+\
                            f'\t\taccuracy: \t{self.best_performance_emo["accuracy"]}\n'+\
                            f'\t\tmacro_f1: \t{self.best_performance_emo["macro_f1"]}\n'+\
                            f'\t\tweighted_f1: \t{self.best_performance_emo["weighted_f1"]}\n'+\
                            f'\t<Emotion-Cause Prediction: [Epoch: {self.best_performance_cau["epoch"]}]>\n'+\
                            f'\t\taccuracy: \t{self.best_performance_cau["accuracy"]}\n'+\
                            f'\t\tprecision: \t{self.best_performance_cau["precision"]}\n'+\
                            f'\t\trecall: \t{self.best_performance_cau["recall"]}\n'+\
                            f'\t\tf1:\t\t{self.best_performance_cau["f1"]}\n'+\
                            f'\t<Pair-Emotion Prediction: [Epoch: {self.best_performance_emo_cau["epoch"]}]>\n'+\
                            f'\t\tprecision: \t{self.best_performance_emo_cau["precision"]}\n'+\
                            f'\t\trecall: \t{self.best_performance_emo_cau["recall"]}\n'+\
                            f'\t\tf1:\t\t{self.best_performance_emo_cau["f1"]}\n'
            
        if (types == 'valid'):
            logging_texts += appended_log_valid
            
        logger.info(logging_texts)
        if (types == 'test'):
            # label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
            label_ = np.array([1, 11, 21, 31, 41, 51, 61, 0, 10, 20, 30, 40, 50, 60])
            confusion_pred = confusion_matrix(torch.cat(self.emo_cause_true_y_list[types]).to('cpu'), torch.cat(self.emo_cause_pred_y_list[types]).to('cpu'), labels=label_)
            confusion_all = confusion_matrix(torch.cat(self.emo_cause_true_y_list_all[types]).to('cpu'), torch.cat(self.emo_cause_pred_y_list_all[types]).to('cpu'), labels=label_)
            
            confusion_log = ""
            confusion_log+="[Confusion_pred matrix]\n"
            confusion_log+='\t'
            for label in label_:
                confusion_log+=f'{label}\t'
            confusion_log+="\n"
            for row, label in zip(confusion_pred, label_):
                confusion_log+=f'{label}\t'
                for col in row:
                    confusion_log+=f'{col}\t'
                confusion_log+="\n"
            confusion_log+="[Confusion_all matrix]\n"
            confusion_log+='\t'
            for label in label_:
                confusion_log+=f'{label}\t'
            confusion_log+="\n"
            for row, label in zip(confusion_all, label_):
                confusion_log+=f'{label}\t'
                for col in row:
                    confusion_log+=f'{col}\t'
                confusion_log+="\n"
            logger.info(confusion_log)
            # logger.info(confusion_log)
            
        
    def get_pair_embedding(self, pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        
        utterance_representation = self.dropout(pooled_output)

        concatenated_embedding = torch.cat((utterance_representation, emotion_prediction, 
                                            speaker_ids.view(-1).unsqueeze(1)), dim=1)
        
        pair_embedding = list()
        for batch in concatenated_embedding.view(batch_size, max_doc_len, -1):
            pair_per_batch = list()
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t])))
            pair_embedding.append(torch.stack(pair_per_batch))

        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)

        return pair_embedding
    
    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        pair_info = []
        for speaker_batch, emotion_batch in zip(speaker_ids.view(batch_size, max_doc_len, -1), emotion_prediction.view(batch_size, max_doc_len, -1)):
            info_pair_per_batch = []
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    speaker_condition = speaker_batch[t] == speaker_batch[end_t]
                    emotion_condition = torch.argmax(
                        emotion_batch[t]) == torch.argmax(emotion_batch[end_t])

                    if speaker_condition and emotion_condition:
                        # if speaker and dominant emotion are same
                        info_pair_per_batch.append(torch.Tensor([1, 0, 0, 0]))
                    elif speaker_condition:
                        # if speaker is same, but dominant emotion is differnt
                        info_pair_per_batch.append(torch.Tensor([0, 1, 0, 0]))
                    elif emotion_condition:
                        # if speaker is differnt, but dominant emotion is same
                        info_pair_per_batch.append(torch.Tensor([0, 0, 1, 0]))
                    else:
                        # if speaker and dominant emotion are differnt
                        info_pair_per_batch.append(torch.Tensor([0, 0, 0, 1]))
            pair_info.append(torch.stack(info_pair_per_batch))

        pair_info = torch.stack(pair_info).to(input_ids.device)

        return pair_info
    
def metrics_report_for_emo_binary(pred_y, true_y, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = ['non-neutral', 'neutral']
    pred_y = [1 if element == 6 else 0 for element in pred_y] # element 6 means neutral.
    true_y = [1 if element == 6 else 0 for element in true_y]

    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)

    
def log_metrics(emo_pred_y_list, emo_true_y_list, 
                cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, cau_pred_y_list_all_windowed,
                emo_cause_pred_y_list, emo_cause_true_y_list, emo_cause_pred_y_list_all, emo_cause_true_y_list_all,
                loss_avg, multiclass_avg_type):
    # <<[[ Emotion 부분 ]]>>
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    # logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    emo_report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    emo_report_str = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=False)
    acc_emo, macro_f1, weighted_f1 = emo_report_dict['accuracy'], emo_report_dict['macro avg']['f1-score'], emo_report_dict['weighted avg']['f1-score']
    emo_metrics = (acc_emo, macro_f1, weighted_f1)
    
    # For binary emotion classification
    # logger.info('\n' + metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list)))
    emo_binary_str = metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), get_dict=False)
    # acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    # logger.info(f'\nemotion (binary): {option} | loss {loss_avg}\n')
    
    # <<[[  Cause 부분  ]]>>
    label_ = np.array(['No Cause', 'Cause'])
    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)
    if 'Cause' in report_dict.keys():   #추가된 부분
        _, p_cau, _, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']
    else:   #추가된 부분
        _, p_cau, _, _ = 0, 0, 0, 0   #추가된 부분
        
    # 원래의 recall (window 상관없이 recall을 구한 원래 recall 측정 메트릭)
    # report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)
    
    class_name = list(label_)
    report_dict = classification_report(torch.cat(cau_true_y_list_all).cpu(), torch.cat(cau_pred_y_list_all_windowed).cpu(), target_names=class_name, zero_division=0, digits=4, output_dict=True)
    
    if 'Cause' in report_dict.keys():   #추가된 부분
        acc_cau, _, r_cau, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']
    else:   #추가된 부분
        acc_cau, _, r_cau, _ = 0, 0, 0, 0   #추가된 부분
        
    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
    
    # label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    label_ = np.array([1, 11, 21, 31, 41, 51, 61, 0, 10, 20, 30, 40, 50, 60])
    confusion_pred = confusion_matrix(torch.cat(emo_cause_true_y_list).to('cpu'), torch.cat(emo_cause_pred_y_list).to('cpu'), labels=label_)
    confusion_all = confusion_matrix(torch.cat(emo_cause_true_y_list_all).to('cpu'), torch.cat(emo_cause_pred_y_list_all).to('cpu'), labels=label_)
    
    # 1, 11, 21, 31, 41, 51 순서로
    idx_list = [0, 1, 2, 3, 4, 5]
    pred_num_acc_list = [0, 0, 0, 0, 0, 0]
    pred_num_acc_list_all = [0, 0, 0, 0, 0, 0]
    pred_num_precision_denominator_dict = [0, 0, 0, 0, 0, 0]
    pred_num_recall_denominator_dict = [0, 0, 0, 0, 0, 0]
    
    for i in idx_list:
        pred_num_acc_list[i] = confusion_pred[i][i]
        pred_num_acc_list_all[i] = confusion_all[i][i]
        pred_num_precision_denominator_dict[i] = sum(confusion_pred[:,i])
        pred_num_recall_denominator_dict[i] = sum(confusion_all[i,:])
    
    # support_ratio = [0, 0, 0, 0, 0, 0]
    # for i in range(len(support_ratio)):
    #     support_ratio[i] = pred_num_recall_denominator_dict[i]/sum(pred_num_recall_denominator_dict)
        
    '''# Confusion matrix 시각화
    print('\t', end="")
    for label in label_:
        print(label, '\t', end="")
    print("")
    for row, label in zip(confusion_pred, label_):
        print(label, '\t', end="")
        for col in row:
            print(col, '\t', end="")
        print("")
        '''
    '''
    <Pair-emotion F1 알고리즘 정리>
    - 각 Pair-Emotion 예측 여부에 따라 7*2 크기의 confusion matrix를 만듦
    - 클래스 1, 11, 21, 31, 41, 51 [(pair,angry), (pair,disgust), (pair,fear), ..., (pair,surprise)]에 대해 각각의 precision, recall 구한다.
    - 구한 각 클래스의 precision, recall을 평균내어 전체 precision, recall 구한다 (macro average)
    - 전체 precision, recall을 통해 Pair-Emotion F1 구한다.
    '''
    # 불균형이 심하므로 micro
    micro_precision = sum(pred_num_acc_list) / sum(pred_num_precision_denominator_dict)
    micro_recall = sum(pred_num_acc_list) / sum(pred_num_recall_denominator_dict) # 정답은 단 한 종류만 허용
    # micro_recall = sum(pred_num_acc_list_all) / sum(pred_num_recall_denominator_dict)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall != 0 else 0
    
    p_emo_cau = micro_precision
    r_emo_cau = micro_recall
    f1_emo_cau = micro_f1
    
    # p_emo_cau /= len(idx_list)
    # r_emo_cau /= len(idx_list)
    # f1_emo_cau = 2 * p_emo_cau * r_emo_cau / (p_emo_cau + r_emo_cau) if p_emo_cau + r_emo_cau != 0 else 0
    
    return emo_report_str, emo_metrics, emo_binary_str, acc_cau, p_cau, r_cau, f1_cau, p_emo_cau, r_emo_cau, f1_emo_cau

def argmax_prediction(pred_y, true_y):
    pred_argmax = torch.argmax(pred_y, dim=1).cpu()
    true_y = true_y.cpu()
    return pred_argmax, true_y

def threshold_prediction(pred_y, true_y):
    pred_y = pred_y > 0.5
    return pred_y, true_y

def metrics_report(pred_y, true_y, label, get_dict=False, multilabel=False):
    true_y = true_y.view(-1)
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = list(label[available_label])
    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)
