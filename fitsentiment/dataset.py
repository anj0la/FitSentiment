import os
import pandas as pd
from torch.utils.data import Dataset

class CustomWorkoutSplitsDataset(Dataset):
    def __init__(self, annotations_file):
        self.text_labels = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.text_labels)

    def __getitem__(self, idx):
        text = self.text_labels.iloc[idx, 0]
        label = self.text_labels.iloc[idx, 1]
        return text, label