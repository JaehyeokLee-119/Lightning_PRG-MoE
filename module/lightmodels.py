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

class EmotionModel(pl.LightningModule):
    def __init__(self, emotion_encoder_name, n_emotion=7, dropout=0.5):
        super().__init__()
        # Model
        self.encoder = AutoModel.from_pretrained(emotion_encoder_name)
        self.emotion_linear = nn.Linear(self.encoder.config.hidden_size, n_emotion)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, max_seq_len):
        _, pooled_output_emotion = self.encoder(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        
        utterance_representation = self.dropout(pooled_output_emotion)
        emotion_prediction = self.emotion_linear(utterance_representation)
        
        return emotion_prediction, pooled_output_emotion

class CauseModel(pl.LightningModule):
    def __init__(self, cause_encoder_name, guiding_lambda, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        # Model
        self.encoder = AutoModel.from_pretrained(cause_encoder_name)
        self.gating_network = nn.Linear(2 * (self.encoder.config.hidden_size + n_emotion + 1), n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(2 * (self.encoder.config.hidden_size + n_emotion + 1), 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        self.guiding_lambda = guiding_lambda
        self.n_expert = n_expert
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        
    def forward(self, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):
        _, pooled_output_cause = self.encoder(input_ids=input_ids.view(-1, max_seq_len),
                                attention_mask=attention_mask.view(-1, max_seq_len),
                                return_dict=False)
        
        pair_embedding = self.get_pair_embedding(pooled_output_cause, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return cause_pred

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
    
class TotalModel(pl.LightningModule):
    def __init__(self, emotion_encoder_name, cause_encoder_name, guiding_lambda, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        # Model
        self.encoder_emotion = AutoModel.from_pretrained(emotion_encoder_name)
        self.emotion_linear = nn.Linear(self.encoder_emotion.config.hidden_size, n_emotion)
        self.encoder_cause = AutoModel.from_pretrained(cause_encoder_name)
        self.gating_network = nn.Linear(2 * (self.encoder_cause.config.hidden_size + n_emotion + 1), n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(2 * (self.encoder_cause.config.hidden_size + n_emotion + 1), 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        self.guiding_lambda = guiding_lambda
        self.n_expert = n_expert
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):
        _, pooled_output_emotion = self.encoder_emotion(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        
        utterance_representation = self.dropout(pooled_output_emotion)
        emotion_prediction = self.emotion_linear(utterance_representation)
        
        _, pooled_output_cause = self.encoder_cause(input_ids=input_ids.view(-1, max_seq_len),
                                attention_mask=attention_mask.view(-1, max_seq_len),
                                return_dict=False)
        
        pair_embedding = self.get_pair_embedding(pooled_output_cause, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return cause_pred

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