import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

class LSTM(nn.Module):
    
    def __init__(self, vocab_size: int, embedding_dim: int, lstm_hidden_dim: int, hidden_dims: list[int], output_dim: int, n_layers: int, dropout: int, batch_first=True, bidirectional=False):
        super(LSTM, self).__init__()
                
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # long-term short memory (LSTM) layer(s)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=n_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
        # multi-layer perceptron
        self.feed_forward = self._create_multi_layer_perceptron(lstm_hidden_dim=lstm_hidden_dim, hidden_dims=hidden_dims, output_layer=output_dim)
        
    def forward(self, input_text, input_text_lengths):
        """
        Implements the forward pass for the CBOW architecture.

        Args:
            x (Tensor): the input to the CBOW model. The shape of the text is [batch size, length].

        Returns:
            Any: the log probability of the model (i.e., the prediction).
        """
        # embeeding shape = [batch size, sent_len, emb dim]
        embeddings = self.embedding(input_text) 
        
        # packed embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=input_text_lengths, batch_first=True)
        
        # getting output from lstm
        packed_output, (hidden, cell) = self.lstm(packed_embeddings)
        
        # getting last hidden state
        last_hidden = hidden[-1]
        
        # passing hidden state through mlp
        output = self.feed_forward(last_hidden)

        return output
        
    def _create_multi_layer_perceptron(self, lstm_hidden_dim: int, hidden_dims: list[int], output_layer: int) -> nn.Sequential:
        layers = []
        # hidden layers
        input_dim = lstm_hidden_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        # output layer
        layers.append(nn.Linear(input_dim, output_layer))
        return nn.Sequential(*layers)

# Usage