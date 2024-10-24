import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(
        self, 
        hid_size, 
        out_slot, 
        out_int, 
        emb_size, 
        vocab_len, 
        pad_index, 
        n_layer=1, 
        bidirectional=False,
        emb_dropout=0.0,
        out_dropout=0.0,
        hid_dropout=0.0
    ):
        """Init model.

        Args:
            hid_size (int): Hidden size.
            out_slot (int): Number of slots (output size for slot filling).
            out_int (int): Number of intents (output size for intent class).
            emb_size (int): Word embedding size.
            vocab_len (int): Vocabulary length.
            pad_index (int): Padding token index.
            n_layer (int, optional): _description_. Defaults to 1.
            bidirectional (bool, optional): _description_. Defaults to False.
        """
        super(ModelIAS, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=bidirectional, batch_first=True)
        
        if bidirectional:    
            self.slot_out = nn.Linear(2*hid_size, out_slot)
        else:
            self.slot_out = nn.Linear(hid_size, out_slot)
        
        self.intent_out = nn.Linear(hid_size, out_int)
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else None
        self.hid_dropout = nn.Dropout(hid_dropout) if hid_dropout > 0 else None
        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else None
        
    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = self.embedding(utterance) 
        if self.emb_dropout:
            utt_emb = self.emb_dropout(utt_emb)
        
        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input) 
       
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        if self.out_dropout:
            self.out_dropout(last_hidden)
        intent = self.intent_out(last_hidden)
        
        # Slot size: batch_size, seq_len, classes 
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent