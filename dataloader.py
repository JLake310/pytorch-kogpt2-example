import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        data = pd.read_csv(file_path)
        concats = [
            label + "|" + text for label, text in zip(data["target"], data["text"])
        ]
        self.item = tokenizer(
            concats,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )["input_ids"]
        self.length = len(concats)

    def __getitem__(self, i):
        return self.item[i]

    def __len__(self):
        return self.length


def GPTDataLoader(tokenizer, file_path, batch_size):
    data = GPTDataset(tokenizer, file_path)
    return DataLoader(data, batch_size=batch_size)
