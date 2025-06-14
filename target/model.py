import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class BERTCRFForNER(nn.Module):
    def __init__(self, model_name, num_labels, lstm_hidden_size=256, lstm_layers=1, dropout_rate=0.1):
        super(BERTCRFForNER, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        lstm_output, _ = self.lstm(sequence_output)  # [B, L, 2*hidden]
        logits = self.classifier(lstm_output)         # [B, L, num_labels]

        if labels is not None:
            # 替换 -100，防止标签越界
            labels = labels.clone()
            labels[labels == -100] = 0
            loss = -self.crf(logits, labels, mask=attention_mask.bool(), reduction='mean')
            return loss
        else:
            pred = self.crf.decode(logits, mask=attention_mask.bool())
            return pred
