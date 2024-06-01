import pandas as pd
import numpy as np
import transformers
import torch
from torch.utils.data import Dataset


class RadiologyLabeledDataset(Dataset):
    def __init__(self, tokenizer, df, max_length, target, view_name):
        super(RadiologyLabeledDataset, self).__init__()
        self.view_name = view_name
        self.df = df
        self.tokenizer = tokenizer
        self.target = self.df.loc[:, target].to_numpy()
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.loc[index, self.view_name]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "target": torch.tensor(self.target[index], dtype=torch.long),
            "file": [self.df.loc[index, "File Name"]],
        }


class RadiologyUnlabeledDataset(Dataset):
    def __init__(self, tokenizer, df, max_length, view_name, other_view_name):
        super(RadiologyUnlabeledDataset, self).__init__()
        self.view = view_name
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.other_view = other_view_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.loc[index, self.view]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )
        ids = inputs["input_ids"]
        # token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "file": [self.df.loc[index, "File Name"]],
            "other_view": [self.df.loc[index, self.other_view]],
        }
