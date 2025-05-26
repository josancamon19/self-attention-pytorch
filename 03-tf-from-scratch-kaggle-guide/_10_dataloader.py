import torch
from torch.utils.data import Dataset


# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html


class TextDataset(Dataset):
    def __init__(self, x_dataframe, y_dataframe):
        self.x_dataframe = x_dataframe
        self.y_dataframe = y_dataframe

    def __len__(self):
        return len(self.x_dataframe)

    def __getitem__(self, idx):
        x = self.x_dataframe.iloc[idx]  # Get the 'tokenized' data
        y = self.y_dataframe.iloc[idx]  # Get the 'target' data
        return torch.LongTensor(x), torch.tensor(y, dtype=torch.float32)
