import torch.nn as nn
from transformers import BertTokenizer, BertModel

class ModelABSA(nn.Module):

    def __init__(
        self,
        tokenizer, 
        bert,
        out_slot,
        dropout=0.0
    ):
        """Init model.

        Args:
            out_slot (int): Number of slots (output size for slot filling).
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            version (str, optional): BERT version name. Defaults to "bert-base-uncased".
        """
        super(ModelABSA, self).__init__()
        self.tokenizer = tokenizer
        self.bert = bert
        
        hid_size = self.bert.config.hidden_size
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.slot_dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, inputs):   
        """Compute forward pass of model.""" 
        #padded_tokens = self.tokenizer(inputs, return_tensors="pt", padding=True)
        ouputs = self.bert(**inputs)
        
        # Hidden representation of each token -> slot filling
        last_hidden_states = ouputs.last_hidden_state
        
        if self.slot_dropout:
            last_hidden_states = self.slot_dropout(last_hidden_states)
        slots = self.slot_out(last_hidden_states).permute(0,2,1)
        
        # Ignore embeddings for [CLS] and [SEP] tokens for slots to match dimensions
        return slots[:,:,1:-1]
        