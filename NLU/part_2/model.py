import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer

class ModelIAS(nn.Module):

    def __init__(
        self,
        out_slot,
        out_intent,
        dropout=0.0,
        version="bert-base-uncased"
    ):
        """Init model.

        Args:
            out_slot (int): Number of slots (output size for slot filling).
            out_intent (int): Number of intents (output size for intent class).
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            version (str, optional): BERT version name. Defaults to "bert-base-uncased".
        """
        super(ModelIAS, self).__init__()
        if version == "bert-tiny-uncased":
            self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
            self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        elif version == "bert-mini-uncased":
            self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
            self.bert = AutoModel.from_pretrained("prajjwal1/bert-mini")
        elif version == "bert-small-uncased":
            self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
            self.bert = AutoModel.from_pretrained("prajjwal1/bert-small")
        elif version == "bert-medium-uncased":
            self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-medium")
            self.bert = AutoModel.from_pretrained("prajjwal1/bert-medium")
        else: 
            self.tokenizer = BertTokenizer.from_pretrained(version)
            self.bert = BertModel.from_pretrained(version)
        
        hid_size = self.bert.config.hidden_size
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_intent)
        self.slot_dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.intent_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, inputs):   
        """Compute forward pass of model.""" 
        padded_tokens = self.tokenizer(inputs, return_tensors="pt", padding=True)
        ouputs = self.bert(**padded_tokens)
        # Hidden representation of [CLS] -> intent
        pooler_out = ouputs.pooler_output 
        
        # Hidden representation of each token -> slot filling
        last_hidden_states = ouputs.last_hidden_state
        
        if self.slot_dropout and self.intent_dropout:
            last_hidden_states = self.slot_dropout(last_hidden_states)
            pooler_out = self.intent_dropout(pooler_out)
        slots = self.slot_out(last_hidden_states).permute(0,2,1)
        intents = self.intent_out(pooler_out)
        
        # Ignore embeddings for [CLS] and [SEP] tokens for slots to match dimensions
        return slots[:,:,1:-1], intents
        