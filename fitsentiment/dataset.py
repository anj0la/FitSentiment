from ast import literal_eval
import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomWorkoutSplitsDataset(Dataset):
    def __init__(self, annotations_file: str) -> None:
        self.text_labels = pd.read_csv(annotations_file)

    def __len__(self) -> int:
        return len(self.text_labels)

    def __getitem__(self, idx: int) -> tuple[list[int], int, int]:
        sequence = literal_eval(self.text_labels.iloc[idx, 0])
        label = self.text_labels.iloc[idx, 1]
        text_length = len(sequence)
        return sequence, label, text_length