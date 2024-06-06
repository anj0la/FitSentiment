"""
File: train.py

Author: Anjola Aina
Date Modified: June 5th, 2024

This file contains all the necessary functions used to train the model.
Only run this file if you want to add more training examples to improve the performance of the model.
Otherwise, use the pretrained model in the 'models' folder, called model_saved_weights.pt.

Functions:
    preprocess_and_save_corpus(str, str) -> dict: Reads a dataset from a CSV file, preprocesses it, and saves it to a new CSV file, returning the vocabulary.
    create_dataloaders(str, int = 64, float = 0.8) -> tuple[DataLoader, DataLoader]: Creates custom datasets and dataloaders for training and testing.
    train(LSTM, DataLoader, optim.SGD) -> tuple[float, float]: Trains the model for one epoch.
    evaluate(LSTM, DataLoader) -> tuple[float, float, float, float, float]: Evaluates the model on the validation/test set.


"""
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from fitsentiment.dataset import CustomWorkoutSplitsDataset
from fitsentiment.preprocess_data import TextPipeline
from fitsentiment.model import LSTM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def preprocess_and_save_corpus(input_file: str, output_file: str) -> dict:
    """
    Reads a dataset from a CSV file, preprocesses the text data, encodes the tokens and labels, 
    saves the processed data to a new CSV file, and returns the vocabulary.

    Args:
        input_file (str): The path to the input CSV file containing the raw corpus.
        output_file (str): The path to save the processed CSV file.

    Returns:
        dict: The vocabulary dictionary.
    """
    # reading the dataset scraped from Reddit to a Pandas DataFrame
    df = pd.read_csv(input_file)

    # preprocessing the corpus and encoding the text and labels
    text_pipeline = TextPipeline()
    encoded_text, encoded_labels = text_pipeline.fit(df)

    # saving the processed data to a new CSV file
    text_pipeline.convert_to_csv(text=encoded_text, labels=encoded_labels, file_path=output_file)
    
    # returning the vocab
    return text_pipeline.vocab

def collate_batch(batch: tuple[list[int], int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of data for the DataLoader.

    This function takes a batch of sequences, labels, and lengths, converts them to tensors, 
    and pads the sequences to ensure they are of equal length. This is useful for feeding data 
    into models that require fixed-length inputs, such as LSTM models.

    Args:
        batch (list of tuples): A list where each element is a tuple containing three elements:
            - sequences (list of int): The sequence of token ids representing a piece of text.
            - labels (int): The label corresponding to the sequence.
            - lengths (int): The original length of the sequence.

    Returns:
        tuple: A tuple containing three elements:
            - padded_sequences (torch.Tensor): A tensor of shape (batch_size, max_sequence_length) 
              containing the padded sequences.
            - labels (torch.Tensor): A tensor of shape (batch_size,) containing the labels.
            - lengths (torch.Tensor): A tensor of shape (batch_size,) containing the original lengths 
              of the sequences.
    """
    sequences, labels, lengths = zip(*batch)
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    labels = torch.tensor(labels, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    # pad sequences
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    return padded_sequences, labels, lengths


def create_dataloaders(file_path: str, batch_size: int = 64, train_split: float = 0.8) -> tuple[DataLoader, DataLoader]:
    """
    Creates custom datasets and dataloaders for training and testing.

    Args:
        file_path (str): The path to the processed CSV file containing the data.
        batch_size (int): The size of the batches for the dataloaders. Default is 64.
        train_split (float): The proportion of the data to use for training. Default is 0.8.

    Returns:
        tuple: A tuple containing:
            - DataLoader: The dataloader for the training dataset.
            - DataLoader: The dataloader for the testing dataset.
    """
    # creating the custom dataset
    dataset = CustomWorkoutSplitsDataset(file_path)

    # splitting the dataset into training and testing sets
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # creating dataloaders for the training and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    return train_dataloader, test_dataloader

def train(model: LSTM, iterator: DataLoader, optimizer: optim.SGD, device: torch.device) -> tuple[float, float]:
    """
    Trains the model for one epoch.

    Args:
        model (LSTM): The model to be trained.
        iterator (DataLoader): The DataLoader containing the training data.
        optimizer (optim.SGD): The optimizer used for updating model parameters.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the epoch.
            - float: The average accuracy over the epoch.
    """
    # initialize the epoch loss and accuracy for every epoch 
    epoch_loss = 0
    epoch_accurary = 0
    
    # set the model in the training phase
    model.train()  
    
    # going through each batch in the training iterator
    for batch in iterator:
        
        # getting the padded sequences, labels and lengths from batch 
        padded_sequences, labels, lengths = batch
        
        # move to GPU
        padded_sequences = padded_sequences.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # resetting the gradients after every batch
        optimizer.zero_grad()   
                
        # getting expected predictions
        predictions = torch.round(model(padded_sequences, lengths).squeeze())
                
        # computing the loss
        loss = F.cross_entropy(predictions, labels)        
        
        # computing metrics 
        accurary = accuracy_score(y_true=labels.detach().numpy(), y_pred=predictions.detach().numpy())   
        
        # backpropagating the loss and computing the gradients
        loss.backward()       
        
        # updating the weights
        optimizer.step()      
        
        # incrementing the loss and accuracy
        epoch_loss += loss.item()  
        epoch_accurary += accurary  
        
    return epoch_loss / len(iterator), epoch_accurary / len(iterator)

def evaluate(model: LSTM, iterator: DataLoader, device: torch.device) -> tuple[float, float, float, float, float]:
    """
    Evaluates the model on the validation/test set.

    Args:
        model (LSTM): The model to be evaluated.
        iterator (DataLoader): The DataLoader containing the validation/test data.

    Returns:
        tuple: A tuple containing:
            - float: The average loss over the validation/test set.
            - float: The average accuracy over the validation/test set.
            - float: The average precision over the validation/test set.
            - float: The average recall over the validation/test set.
            - float: The average F1 score over the validation/test set.
    """
    # initialize the epoch loss and accuracy for every epoch 
    epoch_loss = 0
    epoch_accurary = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_f1_score = 0

    # deactivating dropout layers
    model.eval()
    
    # deactivating autograd
    with torch.no_grad():
        
        for batch in iterator:
            
            # getting the padded sequences, labels and lengths from batch 
            padded_sequences, labels, lengths = batch
            
            # move to GPU
            padded_sequences = padded_sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # getting expected predictions
            predictions = torch.round(model(padded_sequences, lengths).squeeze())
            
            # computing the loss
            loss = F.cross_entropy(predictions, labels)        
            
            # computing metrics 
            accurary = accuracy_score(y_true=labels, y_pred=predictions)
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='weighted', zero_division=1)
            
            # keeping track of metrics
            epoch_loss += loss.item()
            epoch_accurary += accurary
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1_score += f1_score
    
    return epoch_loss / len(iterator), epoch_accurary / len(iterator), epoch_precision / len(iterator), epoch_recall / len(iterator), epoch_f1_score / len(iterator)
        
        
def train_loop(model: LSTM, train_iterator: DataLoader, test_iterator: DataLoader, device: torch.device, n_epochs: int = 10, 
               lr: float = 0.2, weight_decay: float = 0.0, model_save_path: str = 'model/model_saved_weights.pt') -> None:
    """
    Train the model for multiple epochs and evaluate on the validation set.

    Args:
        model (LSTM): The model to be trained.
        train_iterator (DataLoader): The DataLoader containing the training data.
        valid_iterator (DataLoader): The DataLoader containing the validation data.
        n_epochs (int, optional): Number of epochs to train the model (default is 5).
        lr (float, optional): Learning rate for the optimizer (default is 0.01).
        weight_decay (float, optional): Weight decay for regularization (default is 0.0).
        model_save_path (str, optional): Path to save the best model's weights (default is 'model/model_saved_weights.pt').
    """
    best_test_loss = float('inf')
    optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        
        # train the model
        train_loss, train_accurary = train(model, train_iterator, optimizer, device)
        
        # evaluate the model
        test_loss, test_accurary, precision, recall, f1_score = evaluate(model, test_iterator, device)
        
        # save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(obj=model.state_dict(), f=model_save_path)
        
        # printing metrics
        print(f'\t Epoch: {epoch} out of {n_epochs}')
        print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_accurary * 100:.2f}%')
        print(f'\t Valid Loss: {test_loss:.3f} | Valid Acc: {test_accurary * 100:.2f}%')
        print(f'\t Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1_score:.2f}')


##### Running the code
input_file_path = 'data/corpus.csv'
output_file_path = 'data/corpus_clean.csv'

# preprocessing data and getting the vocab
vocab = preprocess_and_save_corpus(input_file=input_file_path, output_file=output_file_path)

# getting the dataloaders
train_dataloader, test_dataloader = create_dataloaders(file_path=output_file_path)

# defining the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# defining the model
model = LSTM(vocab_size=len(vocab)).to(device)

print(model)

# running the train loop
train_loop(model=model, train_iterator=train_dataloader, test_iterator=test_dataloader, device=device)


