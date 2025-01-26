import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class Waterbirds(Dataset):
    """
    CUB dataset (already cropped and centered).
    """
    def __init__(self, root, split, transform, pseudo_bias=None, metadata_csv_name="metadata.csv"):
        assert os.path.exists(root), f"'{root}' does not exist yet. Please generate the dataset first."
        assert split in ["train", "valid", "test"], f"'{split}' is not a valid split"

        self.root = root
        self.transform = transform
        self.split = split
        self.split_dict = {"train": 0, "valid": 1, "test": 2,}

        # read metadata
        self.metadata_df = pd.read_csv(os.path.join(self.root, metadata_csv_name))
        self.metadata_df = self.metadata_df[self.metadata_df["split"] == self.split_dict[self.split]]
        # extract filenames
        self.filename = self.metadata_df["img_filename"].values

        # get the target values
        self.targets = torch.tensor(self.metadata_df["y"].values)
        self.biases = torch.tensor(self.metadata_df["place"].values)
        self.groups = (self.targets * 2 + self.biases).long()

        
        if pseudo_bias is not None:
            self.biases = torch.load(pseudo_bias).numpy()

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.filename[idx])
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            x = self.transform(img)
        
        # TODO: remove never used items
        return {
            "img": x, "label": self.targets[idx],
            "group_label": self.groups[idx], "spurious_label": self.biases[idx],
            "idx": idx, "path": path,
        }
