import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup
from module.evaluation import log_metrics, FocalLoss
from sklearn.metrics import classification_report
from transformers import AutoModel
import numpy as np
from module.lightmodels import TotalModel, OriginalPRG_MoE, TotalModel_cause_fc

class LitPRGMoE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # 모델 셋팅 파라미터
        self.encoder_name = kwargs['encoder_name']
        # Model
        
        self.use_original = kwargs['use_original']
        self.use_newfc = kwargs['use_newfc']
        
        if self.use_original:
            self.model = OriginalPRG_MoE() # output: (emotion prediction, cause prediction)
        else:
            if self.use_newfc:
                self.model = TotalModel_cause_fc(self.encoder_name) # output: (emotion prediction, cause prediction)
            else:
                self.model = TotalModel(self.encoder_name) # output: (emotion prediction, cause prediction)

        # 하이퍼파라미터 설정
        self.training_iter = kwargs['training_iter']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.window_size = 3
        self.n_expert = 4
        self.n_emotion = 7
        self.guiding_lambda = kwargs['guiding_lambda']
        self.loss_lambda = kwargs['loss_lambda'] # loss 중 Emotion loss의 비율
        
        # 학습 방법 설정
        self.num_unfreeze = kwargs['unfreeze']
        self.only_emotion = False
        self.n_cause = kwargs['n_cause']
        self.emotion_epoch_ratio = kwargs['emotion_epoch_ratio'] # 이 비율만큼 추가적으로 먼저 감정만 학습
        self.ckpt_type = kwargs['ckpt_type']
        
        # 모델 내 학습 중 변수 설정
        self.test = False # True when testing(on_test_epoch_start ~ on_test_epoch_end)
            # test에서 joint_accuracy를 계산
        self.train_type = 'total'
        
        if 'bert-base' in self.encoder_name:
            self.is_bert_like = True
        else:
            self.is_bert_like = False
                        
        self.emotion_epoch = int(self.training_iter * self.emotion_epoch_ratio)
        self.training_iter = self.training_iter + self.emotion_epoch
        
        # Dictionaries for logging
        types = ['train', 'valid', 'test']
        self.emo_pred_y_list = {}
        self.emo_true_y_list = {}
        self.cau_pred_y_list = {}
        self.cau_true_y_list = {}
        self.cau_pred_y_list_all = {}
        self.cau_true_y_list_all = {}
        self.loss_sum = {}
        self.batch_count = {}
        
        for i in types:
            self.emo_pred_y_list[i] = []
            self.emo_true_y_list[i] = []
            self.cau_pred_y_list[i] = []
            self.cau_true_y_list[i] = []
            self.cau_pred_y_list_all[i] = []
            self.cau_true_y_list_all[i] = []
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
        self.best_performance_joint = {
            'epoch': 0,
            'loss': 0.0,
            'joint_accuracy': 0.0,
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
        
        if self.test:
            check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=self.window_size, emotion_pred=emotion_prediction)
            check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=1000, )
            self.joint_accuracy_step(emotion_prediction, emotion_label_batch, cause_prediction, pair_binary_cause_label_batch, check_pair_pad_idx, check_pair_window_idx, batch_size, self.n_cause)
        
        return (emotion_prediction, cause_prediction)
    
    def output_processing(self, utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction):
        # 모델의 forward 결과로부터 loss 계산과 로깅을 위한 input 6개를 구해 리턴
        batch_size, _, _ = utterance_input_ids_batch.shape
        # Output processing
        check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=self.window_size, emotion_pred=emotion_prediction)
        check_pair_pad_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=1000, )
        check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.encoder_name)

        # Emotion prediction, label
        emotion_prediction_filtered = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
        emotion_label_batch_filtered = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]
        
        # Cause prediction, label
        pair_binary_cause_prediction_window = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_prediction_all = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_label_batch_all = pair_binary_cause_label_batch[(check_pair_pad_idx != False).nonzero(as_tuple=True)]
        
        return (emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window,
                pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all)
    
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
        (emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window,
                pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        
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
        (emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window,
                pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            
        self.log("valid_loss: ", loss, sync_dist=True)
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
    def test_step(self, batch, batch_idx):
        types = 'test'
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        
        emotion_prediction, binary_cause_prediction = self.forward(batch)
        
        # Output processing
        (emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window,
                pair_binary_cause_prediction_all, pair_binary_cause_label_batch_window, pair_binary_cause_label_batch_all) = self.output_processing(utterance_input_ids_batch, pair_binary_cause_label_batch, emotion_label_batch, emotion_prediction, binary_cause_prediction)
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered, pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            
        self.log("test_loss: ", loss, sync_dist=True)
        # Logging
        self.cau_pred_y_list_all[types].append(pair_binary_cause_prediction_all), self.cau_true_y_list_all[types].append(pair_binary_cause_label_batch_all)
        self.cau_pred_y_list[types].append(pair_binary_cause_prediction_window), self.cau_true_y_list[types].append(pair_binary_cause_label_batch_window)
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        
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
        self.loss_sum[types] = 0.0
        self.batch_count[types] = 0
        
        if self.test == True:
            self.cnt_entire_pair_candidate = 0
            self.cnt_emo_o_pair_o = 0
        
    def log_test_result(self, types='train'):
        logger = logging.getLogger(types)
        
        loss_avg = self.loss_sum[types] / self.batch_count[types]
        emo_report, emo_metrics, acc_cau, p_cau, r_cau, f1_cau = log_metrics(self.train_type, self.emo_pred_y_list[types], self.emo_true_y_list[types], 
                                                self.cau_pred_y_list[types], self.cau_true_y_list[types],
                                                self.cau_pred_y_list_all[types], self.cau_true_y_list_all[types], 
                                                loss_avg)
        
        if types == 'test' or types == 'valid': # joint_accuracy 측정 여부
            if self.cnt_entire_pair_candidate != 0:
                joint_acc = self.cnt_emo_o_pair_o / self.cnt_entire_pair_candidate
                self.log('joint_accuracy', joint_acc, sync_dist=True)
            else:
                joint_acc = 0
                
        self.log('binary_cause 1.loss', loss_avg, sync_dist=True)
        self.log('binary_cause 2.accuracy', acc_cau, sync_dist=True)
        self.log('binary_cause 3.precision', p_cau, sync_dist=True)
        self.log('binary_cause 4.recall', r_cau, sync_dist=True)
        self.log('binary_cause 5.f1-score', f1_cau, sync_dist=True)
        
        self.log('emo 1.accuracy', emo_metrics[0], sync_dist=True)
        self.log('emo 2.macro-f1', emo_metrics[1], sync_dist=True)
        self.log('emo 3.weighted-f1', emo_metrics[2], sync_dist=True)
        
        logging_texts = f'\n[Epoch {self.current_epoch}] / <Emotion Prediction> of {types}\n'+\
                        f'Train type: {self.train_type}\n'+\
                        emo_report+\
                        f'\n<Cause Prediction>'+\
                        f'\n\taccuracy: \t{acc_cau}'+\
                        f'\n\tprecision:\t{p_cau}'+\
                        f'\n\trecall:   \t{r_cau}'+\
                        f'\n\tf1-score: \t{f1_cau}'
                        
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
            if (self.best_performance_joint['joint_accuracy'] < joint_acc):
                self.best_performance_joint['joint_accuracy'] = joint_acc
                self.best_performance_joint['epoch'] = self.current_epoch
                self.best_performance_joint['loss'] = loss_avg
            
            appended_log_valid = f'\nCurrent Best Performance: loss: {self.best_performance_cau["loss"]}\n'+\
                            f'\t<Emotion Prediction: [Epoch: {self.best_performance_emo["epoch"]}]>\n'+\
                            f'\t\taccuracy: \t{self.best_performance_emo["accuracy"]}\n'+\
                            f'\t\tmacro_f1: \t{self.best_performance_emo["macro_f1"]}\n'+\
                            f'\t\tweighted_f1: \t{self.best_performance_emo["weighted_f1"]}\n'+\
                            f'\t<Cause Prediction: [Epoch: {self.best_performance_cau["epoch"]}]>\n'+\
                            f'\t\taccuracy: \t{self.best_performance_cau["accuracy"]}\n'+\
                            f'\t\tprecision: \t{self.best_performance_cau["precision"]}\n'+\
                            f'\t\trecall: \t{self.best_performance_cau["recall"]}\n'+\
                            f'\t\tf1:\t\t{self.best_performance_cau["f1"]}\n'+\
                            f'\t<Joint Accuracy: {self.best_performance_joint["joint_accuracy"]} [Epoch: {self.best_performance_joint["epoch"]}>\n'
            
        if (types == 'test'): # joint_accuracy 측정 여부
            logging_texts += f'\n\tjoint_acc: \t{joint_acc}\n'
            
        if (types == 'valid'):
            logging_texts += appended_log_valid
        logger.info(logging_texts)
    
    def joint_accuracy_step(self, emotion_prediction, emotion_label_batch, binary_cause_prediction, pair_binary_cause_label_batch, check_pair_pad_idx, check_pair_window_idx, batch_size, n_cause):
        # 각 step마다 실행되며, 추론 결과를 바탕으로 self.cnt_entire_pair_candidate, self.cnt_emo_o_pair_o 값을 더해 업데이트한다
        
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
        emotion_pair_pred_expanded = torch.stack(emotion_pair_list).view(batch_size, -1)
        binary_cause_pred_window_full = torch.argmax(binary_cause_prediction.view(batch_size, -1, n_cause), dim=-1)
        emotion_label_pair_list = [] 
        for doc_emotion in emotion_label_batch:
            end_t = 0
            for emotion in doc_emotion:
                for _ in range(end_t+1):
                    emotion_label_pair_list.append(emotion)
                end_t += 1
        emotion_pair_true_expanded = torch.stack(emotion_label_pair_list).view(batch_size, -1)
        pair_label_full = pair_binary_cause_label_batch

        emotion_correct = emotion_pair_pred_expanded == emotion_pair_true_expanded
        emotion_correct_windowed = emotion_correct[(check_pair_window_idx != False).nonzero(as_tuple=True)] # emotion이 정답인 pair들은 True, 아니면 False
        # emotion_correct_all_pad = emotion_correct[(check_pair_pad_idx != False).nonzero(as_tuple=True)] # emotion이 정답인 pair들은 True, 아니면 False
        
        pair_correct = binary_cause_pred_window_full == pair_label_full
        # pair_correct_among_truepair = pair_correct[(pair_label_full == 1)]
        # emotion_correct_among_truepair = emotion_correct[(pair_label_full == 1)]

        pair_correct_windowed = pair_correct[(check_pair_window_idx != False).nonzero(as_tuple=True)] # pair가 정답인 pair들은 True, 아니면 False
        # pair_correct_all_pad = pair_correct[(check_pair_pad_idx != False).nonzero(as_tuple=True)] # pair가 정답인 pair들은 True, 아니면 False
        
        # num_emo_x_pair_o = ((emotion_correct_windowed==False) & (pair_correct_windowed==True)).count_nonzero().item() # : 16
        num_emo_o_pair_o = (emotion_correct_windowed & pair_correct_windowed).count_nonzero().item() # : 28
        # num_emo_x_pair_x = ((emotion_correct_windowed==False) & (pair_correct_windowed==False)).count_nonzero().item()
        # num_emo_o_pair_x = ((emotion_correct_windowed==True) & (pair_correct_windowed==False)).count_nonzero().item()
                
        # num_emo_x_pair_o_all = ((emotion_correct_all_pad==False) & (pair_correct_all_pad==True)).count_nonzero().item() # : 16
        # num_emo_o_pair_o_all = (emotion_correct_all_pad & pair_correct_all_pad).count_nonzero().item() # : 28
        # num_emo_x_pair_x_all = ((emotion_correct_all_pad==False) & (pair_correct_all_pad==False)).count_nonzero().item()
        # num_emo_o_pair_x_all = ((emotion_correct_all_pad==True) & (pair_correct_all_pad==False)).count_nonzero().item()
        
        self.cnt_entire_pair_candidate += len(pair_correct_windowed)              # 5-0) 분류한 emotion에 근거해서, window에 속하므로 정답 pair의 후보가 될 수 있는 utterance pair 개수
        # cnt_correct_pairs += pair_correct_windowed.count_nonzero().item()    # 5-0) 맞춘 pair의 개수 (T인지 F인지)
        # cnt_emo_x_pair_o += num_emo_x_pair_o
        self.cnt_emo_o_pair_o += num_emo_o_pair_o
        # cnt_emo_x_pair_x += num_emo_x_pair_x
        # cnt_emo_o_pair_x += num_emo_o_pair_x
        
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
    
    
def log_metrics(train_type, emo_pred_y_list, emo_true_y_list, cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, loss_avg):
    # train_type = 'cause' / 'emotion' : 리턴값이 다름
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    # logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    emo_report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    emo_report_str = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=False)
    acc_emo, macro_f1, weighted_f1 = emo_report_dict['accuracy'], emo_report_dict['macro avg']['f1-score'], emo_report_dict['weighted avg']['f1-score']
    emo_metrics = (acc_emo, macro_f1, weighted_f1)
    
    # CAUSE 부분
    label_ = np.array(['No Cause', 'Cause'])

    report_dict = metrics_report(torch.cat(cau_pred_y_list), torch.cat(cau_true_y_list), label=label_, get_dict=True)

    if 'Cause' in report_dict.keys():   #추가된 부분
        _, p_cau, _, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']
    else:   #추가된 부분
        _, p_cau, _, _ = 0, 0, 0, 0   #추가된 부분
        
    report_dict = metrics_report(torch.cat(cau_pred_y_list_all), torch.cat(cau_true_y_list_all), label=label_, get_dict=True)
    if 'Cause' in report_dict.keys():   #추가된 부분
        acc_cau, _, r_cau, _ = report_dict['accuracy'], report_dict['Cause']['precision'], report_dict['Cause']['recall'], report_dict['Cause']['f1-score']
    else:   #추가된 부분
        acc_cau, _, r_cau, _ = 0, 0, 0, 0   #추가된 부분
        
    f1_cau = 2 * p_cau * r_cau / (p_cau + r_cau) if p_cau + r_cau != 0 else 0
    # logger.info(f'\nbinary_cause: {option} | loss {loss_avg}\n')
    # logger.info(f'\nbinary_cause: accuracy: {acc_cau} | precision: {p_cau} | recall: {r_cau} | f1-score: {f1_cau}\n')
        
    return emo_report_str, emo_metrics, acc_cau, p_cau, r_cau, f1_cau # For Cause

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
