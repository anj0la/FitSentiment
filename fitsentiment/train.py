import pandas as pd
from torch.utils.data import DataLoader, random_split
from fitsentiment.dataset import CustomWorkoutSplitsDataset
from fitsentiment.preprocess_data import TextPipeline
from fitsentiment.model import LSTM

# reading the dataset scraped from Reddit to a Panadas DataFrame 
df = pd.read_csv('data/corpus.csv')

# preprocessing the corpus and saving it to corpus_clean.csv (put into function)
text_pipeline = TextPipeline()
encoded_text, encoded_labels, vocab = text_pipeline.fit(df)
file_path = 'data/corpus_clean.csv'
processed_df = text_pipeline.convert_to_csv(text=encoded_text, labels=encoded_labels, file_path=file_path)

# creating the custom dataset to load data into batches (putting into own function)
dataset = CustomWorkoutSplitsDataset('data/corpus_clean.csv')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# defining hyperparameters and the model
model = LSTM(vocab_size=len(vocab))

print(model)


