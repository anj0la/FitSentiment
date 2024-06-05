import pandas as pd
from torch.utils.data import DataLoader, random_split
from fitsentiment.dataset import CustomWorkoutSplitsDataset
from fitsentiment.preprocess_data import TextPipeline
from fitsentiment.model import LSTM

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

def create_dataloaders(file_path: str, batch_size: int = 64, train_split: float = 0.8):
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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

# running the code
input_file_path = 'data/corpus.csv'
output_file_path = 'data/corpus_clean.csv'

# preprocessing data and getting the vocab
vocab = preprocess_and_save_corpus(input_file=input_file_path, output_file=output_file_path)

# getting the dataloaders
train_dataloader, test_dataloader = create_dataloaders(file_path=output_file_path)

# defining the model
model = LSTM(vocab_size=len(vocab))

print(model)


