import torch.nn as nn
from transformers import BertTokenizer, BertModel

class ModelIAS(nn.Module):

    def __init__(
        self,
        out_slot,
        out_intent,
        dropout=0.0,
        version="bert-base-uncased"
    ):
        super(ModelIAS, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(version)
        self.bert = BertModel.from_pretrained(version)
        
        hid_size = self.bert.config.hidden_size
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_intent)
        self.slot_dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.intent_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, inputs):    
        padded_tokens = self.tokenizer(inputs, return_tensors="pt", padding=True)
        ouputs = self.bert(**padded_tokens)
        # Hidden representation of [CLS] -> intent
        pooler_out = ouputs.pooler_output 
        
        # Hidden representation of each token -> slot filling
        last_hidden_states = ouputs.last_hidden_state
        
        if self.dropout:
            last_hidden_states = self.slot_dropout(last_hidden_states)
            pooler_out = self.intent_dropout(pooler_out)
        slots = self.slot_out(last_hidden_states).permute(0,2,1)
        intents = self.intent_out(pooler_out)
        
        # Ignore embeddings for [CLS] and [SEP] tokens for slots to match dimensions
        return slots[:,:,1:-1], intents
        