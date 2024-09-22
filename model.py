import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(
        self, 
        rec_layer,
        emb_size, 
        hidden_size, 
        output_size, 
        out_dropout=None,
        emb_dropout=None,
        pad_index=0, 
        n_layers=1
    ):
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(p=emb_dropout) if emb_dropout else None
        
        if rec_layer == "rnn":
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        elif rec_layer == "lstm":
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        else:
            self.rnn = None
            
        self.pad_token = pad_index
        self.out_dropout = nn.Dropout(p=out_dropout) if out_dropout else None
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.emb_dropout:
            emb = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb)
        if self.out_dropout:
            rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0,2,1)
        return output