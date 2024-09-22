import torch.nn as nn

class LM_LSTM_Dropout(nn.Module):
    def __init__(
        self, 
        emb_size, 
        hidden_size, 
        output_size, 
        pad_index=0, 
        out_dropout=0.1,
        emb_dropout=0.1,
        n_layers=1
    ):
        super(LM_LSTM_Dropout, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.out_dropout = nn.Dropout(p=out_dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb_dropout = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb_dropout)
        out_dropout = self.out_dropout(rnn_out)
        output = self.output(out_dropout).permute(0,2,1)
        return output