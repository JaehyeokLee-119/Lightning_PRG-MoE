import lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from module.evaluation import FocalLoss
from module.preprocessing import get_pair_pad_idx, get_pad_idx
from transformers import get_cosine_schedule_with_warmup
from module.evaluation import log_metrics, FocalLoss
from sklearn.metrics import classification_report
from transformers import AutoModel
import numpy as np

class LitPRGMoE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        # Model Setting and Parameters
        self.encoder_name = kwargs['encoder_name']
        self.num_unfreeze = kwargs['unfreeze']
        self.only_emotion = False
        self.n_cause = kwargs['n_cause']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.training_iter = kwargs['training_iter']
        
        self.window_size = 3
        self.n_expert = 4
        self.n_emotion = 7
        self.guiding_lambda = kwargs['guiding_lambda']
        
        if 'bert-base' in self.encoder_name:
            self.is_bert_like = True
        else:
            self.is_bert_like = False
        
        self.train_type = 'cause'
        
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
        
        self.best_performance = 0.0
        # Model
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        self.emotion_linear = nn.Linear(self.encoder.config.hidden_size, self.n_emotion)
        self.gating_network = nn.Linear(2 * (self.encoder.config.hidden_size + self.n_emotion + 1), self.n_expert)
        self.cause_linear = nn.ModuleList()
        for _ in range(self.n_expert):
            self.cause_linear.append(nn.Sequential(nn.Linear(2 * (self.encoder.config.hidden_size + self.n_emotion + 1), 256), nn.Linear(256, self.n_cause)))
        self.dropout = nn.Dropout(self.dropout)
        
        # Model Freeze for only emotion
        if self.only_emotion:
            for name, param in self.gating_network.named_parameters():
                param.requires_grad = False
            for module in self.cause_linear:
                for name, param in module.named_parameters():
                    param.requires_grad = False
                    
    def forward(self, batch):
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = batch
        
        batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
        
        input_ids = utterance_input_ids_batch
        attention_mask = utterance_attention_mask_batch
        token_type_ids = utterance_token_type_ids_batch
        speaker_ids = speaker_batch
        
        _, pooled_output = self.encoder(input_ids=input_ids.view(-1, max_seq_len),
                                    attention_mask=attention_mask.view(-1, max_seq_len),
                                    return_dict=False)
        
        utterance_representation = self.dropout(pooled_output)
        emotion_prediction = self.emotion_linear(utterance_representation)
        
        if self.only_emotion: # only_emotion인 경우는 가짜 아웃풋으로 때움 (cause 부분)
            cause_pred = torch.zeros(int((max_doc_len)*(max_doc_len+1)/2*batch_size)*2).view(-1,2).to(input_ids.device)
        else:  
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
        
        prediction = emotion_prediction, cause_pred
        return prediction
    
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
            criterion_emo = FocalLoss(gamma=2)
            criterion_cau = FocalLoss(gamma=2)
            
            loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
            loss_cau = criterion_cau(pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            loss = 0.2 * loss_emo + 0.8 * loss_cau
        elif self.train_type == 'emotion':
            criterion_emo = FocalLoss(gamma=2)
            loss_emo = criterion_emo(emotion_prediction_filtered, emotion_label_batch_filtered)
            loss = loss_emo
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
        utterance_input_ids_batch, _, _, _, emotion_label_batch, _, pair_binary_cause_label_batch = batch
        
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
        
    def on_train_epoch_end(self):
        # self.loss_avg = self.loss_sum / self.batch_count
        # emo_report, emo_metrics, acc_cau, p_cau, r_cau, f1_cau = self.log_metrics(self.emo_pred_y_list, self.emo_true_y_list, 
        #                                         self.cau_pred_y_list, self.cau_true_y_list,
        #                                         self.cau_pred_y_list_all, self.cau_true_y_list_all, 
        #                                         self.loss_avg)
        # print(emo_report)
        # print(f'cause metrics: acc: {acc_cau}, p: {p_cau}, r: {r_cau}, f1: {f1_cau}')
        self.log_test_result(types='train')
    
    def on_validation_epoch_start(self):
        self.make_test_setting(types='valid')

    def on_validation_epoch_end(self):
        self.log_test_result(types='valid')
    
    def on_test_epoch_start(self):
        self.make_test_setting(types='test')
        
    def on_test_epoch_end(self):
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
        
    def log_test_result(self, types='train'):
        loss_avg = self.loss_sum[types] / self.batch_count[types]
        emo_report, emo_metrics, acc_cau, p_cau, r_cau, f1_cau = log_metrics(self.train_type, self.emo_pred_y_list[types], self.emo_true_y_list[types], 
                                                self.cau_pred_y_list[types], self.cau_true_y_list[types],
                                                self.cau_pred_y_list_all[types], self.cau_true_y_list_all[types], 
                                                loss_avg)
        self.log('binary_cause 1.loss', loss_avg, sync_dist=True)
        self.log('binary_cause 2.accuracy', acc_cau, sync_dist=True)
        self.log('binary_cause 3.precision', p_cau, sync_dist=True)
        self.log('binary_cause 4.recall', r_cau, sync_dist=True)
        self.log('binary_cause 5.f1-score', f1_cau, sync_dist=True)
        
        self.log('emo 1.accuracy', emo_metrics[0], sync_dist=True)
        self.log('emo 2.precision', emo_metrics[1], sync_dist=True)
        self.log('emo 3.recall', emo_metrics[2], sync_dist=True)
        self.log('emo 4.f1-score', emo_metrics[3], sync_dist=True)
        print(f'\n<Emotion Prediction> of {types} / Epoch {self.current_epoch}')
        print(emo_report)
        print(f'<Cause Prediction>\n\taccuracy: \t{acc_cau}\n\tprecision: \t{p_cau}\n\trecall: \t{r_cau}\n\tf1-score: \t{f1_cau}\n')
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=5,
                                                    num_training_steps=self.training_iter,
                                                    )
        return [optimizer], [scheduler]
        
        
    def get_pair_embedding(self, pooled_output, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape

        # # 이 부분 encoder 안 돌리게 최적화 가능한가?
        # _, pooled_output = self.encoder(input_ids=input_ids.view(-1, max_seq_len), 
        #                                 attention_mask=attention_mask.view(-1, max_seq_len), 
        #                                 token_type_ids=token_type_ids.view(-1, max_seq_len), 
        #                                 return_dict=False)
        
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
    
    
def log_metrics(train_type, emo_pred_y_list, emo_true_y_list, cau_pred_y_list, cau_true_y_list, cau_pred_y_list_all, cau_true_y_list_all, loss_avg):
    # train_type = 'cause' / 'emotion' : 리턴값이 다름
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    # logger.info('\n' + metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    emo_report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    emo_report_str = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=False)
    acc_emo, p_emo, r_emo, f1_emo = emo_report_dict['accuracy'], emo_report_dict['weighted avg']['precision'], emo_report_dict['weighted avg']['recall'], emo_report_dict['weighted avg']['f1-score']
    emo_metrics = (acc_emo, p_emo, r_emo, f1_emo)
    
    only_emo_acc, only_emo_macro, only_emo_weighted = acc_emo, emo_report_dict['macro avg']['f1-score'], f1_emo
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
        
    if train_type == 'emotion':
        return only_emo_acc, only_emo_macro, only_emo_weighted # For only emotion
    elif train_type == 'cause':
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
