from transformers import Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
from module.preprocessing import get_data, tokenize_conversation, get_pad_idx, get_pair_pad_idx
from module.evaluation import log_metrics, FocalLoss
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_file, max_seq_len, encoder_name='roberta-large', contain_context=False):
        super().__init__()
        device = "cuda:0"
        data = get_data(dataset_file, device, max_seq_len, encoder_name, contain_context)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = data[0]
        speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = data[1:]
        self.tensor_dataset = (utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        # idx번째 데이터 반환
        # return tuple(tensor[idx] for tensor in self.tensor_dataset)
        data = tuple([self.tensor_dataset[i][idx] for i in range(len(self.tensor_dataset))])
        return data
    
class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, data_collator=None, 
                 tokenizer=None, compute_metrics=None, callbacks=None, optimizers=(None, None), n_cause=2, encoder_name='roberta-large', train_type='cause'):
        super().__init__(model, args, train_dataset, eval_dataset, data_collator, tokenizer, 
                         compute_metrics, callbacks)
        self.n_cause = n_cause
        self.encoder_name = encoder_name
        self.type = train_type
        
    def compute_loss(self, model, inputs, return_outputs=False):
        utterance_input_ids_batch, utterance_attention_mask_batch, utterance_token_type_ids_batch, speaker_batch, emotion_label_batch, pair_cause_label_batch, pair_binary_cause_label_batch = inputs
        batch_size, max_doc_len, max_seq_len = utterance_input_ids_batch.shape
                
        # Get Model output  
        outputs = model(
                        utterance_input_ids_batch, 
                        utterance_attention_mask_batch, 
                        utterance_token_type_ids_batch, 
                        speaker_batch
                        )
        
        emotion_prediction, binary_cause_prediction = outputs
                
        # Output processing
        check_pair_window_idx = get_pair_pad_idx(utterance_input_ids_batch, self.encoder_name, window_constraint=3, emotion_pred=emotion_prediction)
        check_pad_idx = get_pad_idx(utterance_input_ids_batch, self.encoder_name)

        # Emotion prediction, label
        emotion_prediction = emotion_prediction[(check_pad_idx != False).nonzero(as_tuple=True)]
        emotion_label_batch = emotion_label_batch.view(-1)[(check_pad_idx != False).nonzero(as_tuple=True)]
        
        # Cause prediction, label
        pair_binary_cause_prediction_window = binary_cause_prediction.view(batch_size, -1, self.n_cause)[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        pair_binary_cause_label_batch_window = pair_binary_cause_label_batch[(check_pair_window_idx != False).nonzero(as_tuple=True)]
        
        # Loss Calculation
        if self.type == 'cause':
            criterion_emo = FocalLoss(gamma=2)
            criterion_cau = FocalLoss(gamma=2)
            
            loss_emo = criterion_emo(emotion_prediction, emotion_label_batch)
            if (torch.sum(check_pair_window_idx)==0):
                loss_cau = torch.tensor(0.0)
            else:
                loss_cau = criterion_cau(pair_binary_cause_prediction_window, pair_binary_cause_label_batch_window)
            loss = 0.2 * loss_emo + 0.8 * loss_cau
        elif self.type == 'emotion':
            criterion_emo = FocalLoss(gamma=2)
            loss_emo = criterion_emo(emotion_prediction, emotion_label_batch)
            loss = loss_emo
            
        return (loss, outputs) if return_outputs else loss