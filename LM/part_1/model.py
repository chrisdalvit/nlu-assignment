import torch.nn as nn

class LM_RNN(nn.Module):
    """PyTorch model for language modelling."""
    
    def __init__(
        self, 
        rec_layer,
        emb_size, 
        hidden_size, 
        output_size, 
        out_dropout=0.0,
        emb_dropout=0.0,
        pad_index=0, 
        n_layers=1
    ):
        """Init model.

        Args:
            rec_layer (str): Type of recurrent cell. One of ['rnn', 'lstm'].
            emb_size (int): Size of the embedding layer.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer.
            out_dropout (float, optional): Dropout rate for the output layer. Defaults to 0.0 (no dropout).
            emb_dropout (float, optional): Dropout rate for the embedding layer. Defaults to 0.0 (no dropout).
            pad_index (int, optional): Index of the padding token. Defaults to 0.
            n_layers (int, optional): Number of recurrent layers. Defaults to 1.
        """
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(p=emb_dropout) if emb_dropout > 0.0 else None
        
        if rec_layer == "rnn":
            self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        elif rec_layer == "lstm":
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        else:
            self.rnn = None
            
        self.pad_token = pad_index
        self.out_dropout = nn.Dropout(p=out_dropout) if out_dropout > 0.0 else None
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        """Compute forward pass of model."""
        emb = self.embedding(input_sequence)
        if self.emb_dropout:
            emb = self.emb_dropout(emb)
        rnn_out, _  = self.rnn(emb)
        if self.out_dropout:
            rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0,2,1)
        return output