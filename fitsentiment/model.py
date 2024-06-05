"""
File: model.py

Author: Anjola Aina
Date Modified: June 5th, 2024

This module defines an LSTM model for natural language processing tasks using PyTorch.

Classes:
    LSTM: A class that implements an LSTM-based neural network for text classification.

Functions:
    __init__: Initializes the LSTM model with embedding, LSTM, and MLP layers.
    forward: Implements the forward pass of the model.
    _create_multi_layer_perceptron: Creates a multi-layer perceptron for the model.
"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    An LSTM-based model for text classification.

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimensionality of the embeddings. Default is 100.
        lstm_hidden_dim (int): The number of features in the hidden state of the LSTM. Default is 256.
        hidden_dims (list[int]): A list of integers representing the sizes of hidden layers in the MLP. Default is [128, 64, 32].
        output_dim (int): The size of the output layer. Default is 1.
        n_layers (int): The number of recurrent layers in the LSTM. Default is 2.
        dropout (float): The dropout probability. Default is 0.2.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). Default is True.
        bidirectional (bool): If True, becomes a bidirectional LSTM. Default is True.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, lstm_hidden_dim: int = 256, hidden_dims: list[int] = [128, 64, 32], output_dim: int = 1, n_layers: int = 2, dropout: int = 0.2, batch_first: bool = True, bidirectional = True):
        super(LSTM, self).__init__()
                
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # long-term short memory (LSTM) layer(s)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=n_layers, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
        # multi-layer perceptron
        self.feed_forward = self._create_multi_layer_perceptron(lstm_hidden_dim=lstm_hidden_dim, hidden_dims=hidden_dims, output_layer=output_dim)
        
    def forward(self, input_text, input_text_lengths):
        """
        Implements the forward pass for the LSTM model.

        Args:
            input_text (torch.Tensor): The input tensor containing the text data. Shape: [batch size, sequence length].
            input_text_lengths (torch.Tensor): A tensor containing the lengths of each sequence in the batch.

        Returns:
            torch.Tensor: The output of the model after passing through the LSTM and MLP layers.
        """
        # embeeding layer
        embeddings = self.embedding(input_text) 
        
        # packed embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(input=embeddings, lengths=input_text_lengths, batch_first=True)
        
        # lstm layer
        packed_output, (hidden, cell) = self.lstm(packed_embeddings)
        
        # concatenating the last forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # mlp layer (output is one neuron)
        output = self.feed_forward(hidden)

        return output
        
    def _create_multi_layer_perceptron(self, lstm_hidden_dim: int, hidden_dims: list[int], output_layer: int) -> nn.Sequential:
        """
        Creates a multi-layer perceptron (MLP) for the model.

        Args:
            lstm_hidden_dim (int): The number of features in the hidden state of the LSTM.
            hidden_dims (list[int]): A list of integers representing the sizes of hidden layers in the MLP.
            output_layer (int): The size of the output layer.

        Returns:
            nn.Sequential: A sequential container of the MLP layers.
        """
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