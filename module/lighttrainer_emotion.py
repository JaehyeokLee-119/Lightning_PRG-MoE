import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report
from transformers import AutoModel
import numpy as np
from module.lightmodels import EmotionModel, CauseModel

class LitEmotion(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Model Setting and Parameters
        self.emotion_encoder_name = kwargs['emotion_encoder_name']
        self.num_unfreeze = kwargs['unfreeze']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.training_iter = kwargs['training_iter']
        
        self.n_emotion = 7
        self.train_type = 'Emotion_Only'
        
        if 'bert-base' in self.emotion_encoder_name:
            self.is_bert_like = True
        else:
            self.is_bert_like = False
        
        self.best_performance = 0
        
        # Model
        self.emotion_model = EmotionModel(self.emotion_encoder_name, self.n_emotion, self.dropout)
        
        # Logging
        types = ['train', 'valid', 'test']
        self.emo_pred_y_list = {}
        self.emo_true_y_list = {}
        self.loss_sum = {}
        self.batch_count = {}   
        
        
        for i in types:
            self.emo_pred_y_list[i] = []
            self.emo_true_y_list[i] = []
            self.loss_sum[i] = 0.0
            self.batch_count[i] = 0
            
            
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
        
        # emotion forward
        emotion_prediction, pooled_output_emotion = self.emotion_model(input_ids, attention_mask, max_seq_len)
        
        return emotion_prediction, pooled_output_emotion
    
    def output_processing(self, utterance_input_ids_batch, emotion_prediction, emotion_label_batch):
        # 모델의 forward 결과로부터 loss 계산과 로깅을 위한 input 6개를 구해 리턴
        batch_size, _, _ = utterance_input_ids_batch.shape
        # Output processing
        check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.emotion_encoder_name)

        # Emotion prediction, label
        emotion_prediction_filtered = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
        emotion_label_batch_filtered = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]
        
        return (emotion_prediction_filtered, emotion_label_batch_filtered)
    
    def loss_calculation(self, emotion_prediction_filtered, emotion_label_batch_filtered):
        criterion_emo = FocalLoss(gamma=2)
        loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
        return loss_emo
    
    def training_step(self, batch, batch_idx):
        types = 'train'
        # utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, _ = batch
        
        emotion_prediction, pooled_output_emotion = self.forward(batch)
        
        # Output processing (filtering)
        (emotion_prediction_filtered, emotion_label_batch_filtered) = self.output_processing(utterance_input_ids_batch, emotion_prediction, emotion_label_batch)
        
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
        # Logging
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        types = 'valid'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, _ = batch
        
        emotion_prediction, pooled_output_emotion = self.forward(batch)
        
        # Output processing (filtering)
        (emotion_prediction_filtered, emotion_label_batch_filtered) = self.output_processing(utterance_input_ids_batch, emotion_prediction, emotion_label_batch)
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
        # Logging
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        
    def test_step(self, batch, batch_idx):
        types = 'test'
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, _ = batch
        
        emotion_prediction, pooled_output_emotion = self.forward(batch)
        
        # Output processing (filtering)
        (emotion_prediction_filtered, emotion_label_batch_filtered) = self.output_processing(utterance_input_ids_batch, emotion_prediction, emotion_label_batch)
        
        # Loss Calculation
        loss = self.loss_calculation(emotion_prediction_filtered, emotion_label_batch_filtered)
        
        self.loss_sum[types] += loss.item()
        self.batch_count[types] += 1
        
        # Logging
        self.emo_pred_y_list[types].append(emotion_prediction_filtered), self.emo_true_y_list[types].append(emotion_label_batch_filtered)
        

    def on_train_epoch_start(self):
        self.make_test_setting(types='train')
        
    def on_train_epoch_end(self):
        self.log_test_result(types='train')
    
    def on_validation_epoch_start(self):
        self.make_test_setting(types='valid')

    def on_validation_epoch_end(self):
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
        self.loss_sum[types] = 0.0
        self.batch_count[types] = 0
        
    def log_test_result(self, types='train'):
        logger = logging.getLogger(types)
        
        loss_avg = self.loss_sum[types] / self.batch_count[types]
        emo_report, emo_metrics = log_metrics(self.emo_pred_y_list[types], self.emo_true_y_list[types], loss_avg)
        
        self.log('emo 1.accuracy', emo_metrics[0], sync_dist=True)
        self.log('emo 2.precision', emo_metrics[1], sync_dist=True)
        self.log('emo 3.recall', emo_metrics[2], sync_dist=True)
        self.log('emo 4.f1-score', emo_metrics[3], sync_dist=True)
            
        logging_texts = f'\n[Epoch {self.current_epoch}] / <Emotion Prediction> of {types}\n'+\
                        f'Train type: {self.train_type}\n'+\
                        emo_report+'\n'
            
        logger.info(logging_texts)
        
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
                    # backward 시, cycle이 생겨 문제가 생길 경우, batch[end_t].detach() 시도.
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
    
    
def log_metrics(emo_pred_y_list, emo_true_y_list, loss_avg):
    # train_type = 'cause' / 'emotion' : 리턴값이 다름
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    # logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    emo_report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    emo_report_str = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=False)
    acc_emo, p_emo, r_emo, f1_emo = emo_report_dict['accuracy'], emo_report_dict['weighted avg']['precision'], emo_report_dict['weighted avg']['recall'], emo_report_dict['weighted avg']['f1-score']
    emo_metrics = (acc_emo, p_emo, r_emo, f1_emo)
    
    return emo_report_str, emo_metrics

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
