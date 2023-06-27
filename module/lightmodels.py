import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, BertModel, AutoTokenizer
import numpy as np

class TotalModel_exp5(pl.LightningModule):
    def __init__(self, encoder_name, freeze_ratio=0.0, guiding_lambda=0.6, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        self.guiding_lambda = guiding_lambda
        self.n_expert = n_expert+1
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(encoder_name, output_hidden_states=True, num_labels=n_emotion)
        
        tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)
        tokens = ["[Speaker A]", "[Speaker B]", "[/Speaker A]", "[/Speaker B]"]
        tokenizer_.add_tokens(tokens, special_tokens=True)
        
        self.model.resize_token_embeddings(len(tokenizer_))
        # Model freeze
        if freeze_ratio > 0:
            for name, param in self.model.named_parameters():
                for i in range(int(self.model.config.num_hidden_layers * freeze_ratio)):
                    if str(i) in name:
                        param.requires_grad = False
                    
        pair_embedding_size = 2 * (self.model.config.hidden_size + n_emotion + 1)
        
        self.gating_network = nn.Linear(pair_embedding_size, self.n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(self.n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(pair_embedding_size, 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):

        outputs = self.model(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        pooled_output = outputs[1][-1][:,0,:]
        emotion_prediction = outputs[0]
        
        # encoder(AutoModelForClassification)에서 분류 결과와 pooled_output을 받아와야 함
        
        pair_embedding = self.get_pair_embedding(pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return emotion_prediction, cause_pred

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
                    # 쌍을 이루는 두 문장이 같은 문장인지
                    # t == end_t
                    utterance_condition = (t == end_t)
                    speaker_condition = speaker_batch[t] == speaker_batch[end_t]
                    emotion_condition = torch.argmax(
                        emotion_batch[t]) == torch.argmax(emotion_batch[end_t])

                    if utterance_condition:
                            # if the utterances are same
                        info_pair_per_batch.append(torch.Tensor([1, 0, 0, 0, 0]))
                    else: 
                        if speaker_condition and emotion_condition:
                            # if speaker and dominant emotion are same
                            info_pair_per_batch.append(torch.Tensor([0, 1, 0, 0, 0]))
                        elif speaker_condition:
                            # if speaker is same, but dominant emotion is differnt
                            info_pair_per_batch.append(torch.Tensor([0, 0, 1, 0, 0]))
                        elif emotion_condition:
                            # if speaker is differnt, but dominant emotion is same
                            info_pair_per_batch.append(torch.Tensor([0, 0, 0, 1, 0]))
                        else:
                            # if speaker and dominant emotion are differnt
                            info_pair_per_batch.append(torch.Tensor([0, 0, 0, 0, 1]))
            pair_info.append(torch.stack(info_pair_per_batch))

        pair_info = torch.stack(pair_info).to(input_ids.device)

        return pair_info


class TotalModel_exp12(pl.LightningModule):
    def __init__(self, encoder_name, freeze_ratio=0.0, guiding_lambda=0.6, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        self.guiding_lambda = guiding_lambda
        self.n_expert = n_emotion*2
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(encoder_name, output_hidden_states=True, num_labels=n_emotion)
        
        tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)
        tokens = ["[Speaker A]", "[Speaker B]", "[/Speaker A]", "[/Speaker B]"]
        tokenizer_.add_tokens(tokens, special_tokens=True)
        
        self.model.resize_token_embeddings(len(tokenizer_))
        # Model freeze
        if freeze_ratio > 0:
            for name, param in self.model.named_parameters():
                for i in range(int(self.model.config.num_hidden_layers * freeze_ratio)):
                    if str(i) in name:
                        param.requires_grad = False
                    
        pair_embedding_size = 2 * (self.model.config.hidden_size + n_emotion + 1)
        
        self.gating_network = nn.Linear(pair_embedding_size, self.n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(self.n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(pair_embedding_size, 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):

        outputs = self.model(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        pooled_output = outputs[1][-1][:,0,:]
        emotion_prediction = outputs[0]
        
        # encoder(AutoModelForClassification)에서 분류 결과와 pooled_output을 받아와야 함
        
        pair_embedding = self.get_pair_embedding(pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return emotion_prediction, cause_pred

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
                    speaker_condition = speaker_batch[t] == speaker_batch[end_t] # speaker가 같은지
                    # 00,   10,     20,     30,     40,     50,     01,     11,     21,     31,     41,     51
                    # n_expert = 12
                    # 0     1       2       3       4       5       6       7       8       9       10      11
                    one_hot_index = int(speaker_condition*(self.n_expert/2) + torch.argmax(emotion_batch[end_t]))
                    emotion_condition = torch.argmax(
                        emotion_batch[t]) == torch.argmax(emotion_batch[end_t])
                    one_hot_position = torch.Tensor([0] * self.n_expert)
                    one_hot_position[one_hot_index] = 1
                    info_pair_per_batch.append(one_hot_position)
            pair_info.append(torch.stack(info_pair_per_batch))

        pair_info = torch.stack(pair_info).to(input_ids.device)

        return pair_info


class TotalModel(pl.LightningModule):
    def __init__(self, encoder_name, freeze_ratio=0.0, guiding_lambda=0.6, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        self.guiding_lambda = guiding_lambda
        self.n_expert = n_expert
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        
        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(encoder_name, output_hidden_states=True, num_labels=n_emotion)
        
        tokenizer_ = AutoTokenizer.from_pretrained(encoder_name)
        tokens = ["[Speaker A]", "[Speaker B]", "[/Speaker A]", "[/Speaker B]"]
        tokenizer_.add_tokens(tokens, special_tokens=True)
        
        self.model.resize_token_embeddings(len(tokenizer_))
        # Model freeze
        if freeze_ratio > 0:
            for name, param in self.model.named_parameters():
                for i in range(int(self.model.config.num_hidden_layers * freeze_ratio)):
                    if str(i) in name:
                        param.requires_grad = False
                    
        pair_embedding_size = 2 * (self.model.config.hidden_size + n_emotion + 1)
        
        self.gating_network = nn.Linear(pair_embedding_size, n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(pair_embedding_size, 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):
        outputs = self.model(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        pooled_output = outputs[1][-1][:,0,:]
        emotion_prediction = outputs[0]
        
        # encoder(AutoModelForClassification)에서 분류 결과와 pooled_output을 받아와야 함
        
        pair_embedding = self.get_pair_embedding(pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_pred = sum(pred)
        return emotion_prediction, cause_pred

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

class OriginalPRG_MoE(pl.LightningModule):
    def __init__(self, guiding_lambda=0.6, n_emotion=7, n_expert=4, n_cause=2, dropout=0.5):
        super().__init__()
        # Model
        self.encoder = BertModel.from_pretrained('bert-base-cased')
        self.emotion_linear = nn.Linear(self.encoder.config.hidden_size, n_emotion)
        
        self.gating_network = nn.Linear(2 * (self.encoder.config.hidden_size + n_emotion + 1), n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(2 * (self.encoder.config.hidden_size + n_emotion + 1), 256), nn.Linear(256, n_cause)))
        self.dropout = nn.Dropout(dropout)
        self.guiding_lambda = guiding_lambda
        self.n_expert = n_expert
        self.n_emotion = n_emotion
        self.n_cause = n_cause
        print("Original model(PRG-MoE) Loaded")
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids, max_seq_len):
        
        _, pooled_output = self.encoder(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        utterance_representation = self.dropout(pooled_output)
        emotion_prediction = self.emotion_linear(utterance_representation)
        # _, pooled_output
        
        # encoder(AutoModelForClassification)에서 분류 결과와 pooled_output을 받아와야 함
        
        pair_embedding = self.get_pair_embedding(pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        gating_prob = self.guiding_lambda * self.get_subtask_label(
            input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        pred = []
        for _ in range(self.n_expert):
            expert_pred = self.cause_linear[_](pair_embedding.view(-1, pair_embedding.shape[-1]))
            expert_pred *= gating_prob.view(-1,self.n_expert)[:, _].unsqueeze(-1)
            pred.append(expert_pred)

        cause_prediction = sum(pred)
        return emotion_prediction, cause_prediction

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
