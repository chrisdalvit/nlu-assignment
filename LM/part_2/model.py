import torch
import torch.nn as nn

class VariationalDropout(nn.Module):
    """Class for variational dropout."""
    
    def __init__(self, p):
        """Init dropout module.

        Args:
            p (float): Dropout rate.
        """
        super(VariationalDropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        """Compute forward pass of module."""
        if not self.training:
            return x
        mask = torch.ones(x.size(1), x.size(2)).bernoulli(1-self.p).to(x.device)
        mask = mask.expand_as(x)
        return (mask * x) / (1 - self.p)
        
        

class LM_RNN(nn.Module):
    """PyTorch model for language modelling."""
    
    def __init__(
        self, 
        rec_layer,
        emb_size, 
        output_size, 
        out_dropout=0.0,
        emb_dropout=0.0,
        hid_dropout=0.0,
        pad_index=0, 
        n_layers=1,
        weight_tying=False,
        variational_dropout=False
    ):
        """Init model.

        Args:
            rec_layer (str): Type of recurrent cell. One of ['rnn', 'lstm'].
            emb_size (int): Size of the embedding layer.
            output_size (int): Size of the output layer.
            out_dropout (float, optional): Dropout rate for the output layer. Defaults to 0.0 (no dropout).
            emb_dropout (float, optional): Dropout rate for the embedding layer. Defaults to 0.0 (no dropout).
            hid_dropout (float, optional): Dropout rate for the hidden layer. Defaults to 0.0 (no dropout).
            pad_index (int, optional): Index of the padding token. Defaults to 0.
            n_layers (int, optional): Number of recurrent layers. Defaults to 1.
            weight_tying (bool, optional): Activate weight tying. Defaults to False.
            variational_dropout (bool, optional): Use variational dropout. Defaults to False.
        """
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        
        if variational_dropout and emb_dropout > 0.0:
            self.emb_dropout = VariationalDropout(p=emb_dropout)
        elif emb_dropout > 0.0:
            self.emb_dropout = nn.Dropout(p=emb_dropout)
        else:
            self.emb_dropout = None
            
        if variational_dropout and hid_dropout > 0.0:
            self.hid_dropout = VariationalDropout(p=hid_dropout)
        elif hid_dropout > 0.0:
            self.hid_dropout = nn.Dropout(p=hid_dropout)
        else:
            self.hid_dropout = None
           
        if rec_layer == "rnn":
            stacked_rnn = [ nn.RNN(emb_size, emb_size, batch_first=True) for _ in range(n_layers) ]
        elif rec_layer == "lstm":
            stacked_rnn = [ nn.LSTM(emb_size, emb_size, batch_first=True) for _ in range(n_layers) ]
        else:
            stacked_rnn = None
        self.rnns = nn.ModuleList(stacked_rnn)
        
        if variational_dropout and out_dropout > 0.0:
            self.out_dropout = VariationalDropout(p=out_dropout)
        elif out_dropout > 0.0:
            self.out_dropout = nn.Dropout(p=out_dropout)
        else:
            self.out_dropout = None
        
        self.output = nn.Linear(emb_size, output_size)
        if weight_tying:
            self.output.weight = self.embedding.weight
            

    def forward(self, input_sequence):
        """Compute forward pass of model."""
        emb = self.embedding(input_sequence)
        if self.emb_dropout:
            emb = self.emb_dropout(emb)
        
        n_layers = len(self.rnns)
        rnn_out = emb
        for idx, rnn in enumerate(self.rnns):
            rnn.flatten_parameters() # for compact memory usage
            rnn_out, _  = rnn(rnn_out)
            if self.hid_dropout and idx < n_layers:
                rnn_out = self.hid_dropout(rnn_out)
        
        if self.out_dropout:
            rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0,2,1)
        return output