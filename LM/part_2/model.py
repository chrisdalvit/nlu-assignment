import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(
        self, 
        emb_size, 
        hidden_size, 
        output_size, 
        out_dropout=0.0,
        emb_dropout=0.0,
        pad_index=0, 
        n_layers=1,
        weight_tying=False
    ):
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(p=emb_dropout) if emb_dropout > 0.0 else None
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.out_dropout = nn.Dropout(p=out_dropout) if out_dropout > 0.0 else None
        self.output = nn.Linear(hidden_size, output_size)
        if weight_tying:
            self.output.weight = self.embedding.weight
            

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        if self.emb_dropout:
            emb = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb)
        if self.out_dropout:
            rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0,2,1)
        return output