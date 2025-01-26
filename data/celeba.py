import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# TODO: use this as the one and true celebA dataset
class CelebA(datasets.CelebA):
    # TODO NOTE: removed target_attr and bias_attrb
    def __init__(self, root, split, transform, pseudo_bias=None):
        """
        Inputs:
        * `pseudo_bias`: Custom array of bias labels (e.g., labels predicted by a model)
        """
        super(CelebA, self).__init__(root, split, transform=transform, download=False)

        target_idx = self.attr_names.index("Blond_Hair")
        bias_idx = self.attr_names.index("Male")

        self.targets = self.attr[:, target_idx]
        self.biases = self.attr[:, bias_idx]
        self.groups = (self.targets * 2 + self.biases).long()

        if pseudo_bias is not None:
            self.biases = torch.load(pseudo_bias)        
    
    def __getitem__(self, idx):
        path = os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[idx])
        img = Image.open(path)
        
        if self.transform is not None:
            x = self.transform(img)
            
        return {
            "img": x, "label": self.targets[idx],
            "group_label": self.groups[idx], "spurious_label": self.biases[idx],
            "idx": idx, "path": path,
        }
