import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100

class CIFARDataset(Dataset):
    def __init__(self, dataset_name="cifar10", root="./data", train=True, n_shot=None, transform=None):
        self.transform = transform
        
        if dataset_name == "cifar10":
            self.dataset = CIFAR10(root=root, train=train, download=True)
        elif dataset_name == "cifar100":
            self.dataset = CIFAR100(root=root, train=train, download=True)
        
        self.image_list = self.dataset.data
        self.label_list = self.dataset.targets
        self.classes = self.dataset.classes
        
        if n_shot is not None:
            few_shot_samples = []
            c_range = len(self.classes)
            for c in range(c_range):
                c_idx = [idx for idx, label in enumerate(self.label_list) if label == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, min(n_shot, len(c_idx))))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.fromarray(self.image_list[idx]).convert("RGB")
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()

    def get_images_with_label(self, label):
        indices = [idx for idx, lbl in enumerate(self.label_list) if lbl == label]
        images = []
        for idx in indices:
            image = Image.fromarray(self.image_list[idx]).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return images
    
class BaseJsonDataset(Dataset):
    def __init__(self, image_path, json_path, mode='train', n_shot=None, transform=None, dino_transform=None):
        self.transform = transform
        self.dino_transform = dino_transform
        self.image_path = image_path
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        with open(self.split_json) as fp:
            splits = json.load(fp)
            samples = splits[self.mode]
            for s in samples:
                self.image_list.append(s[0])
                self.label_list.append(s[1])
    
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_path, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            default_image = self.transform(image)

        if self.dino_transform:
            dino_image = self.dino_transform(image)
            return default_image, torch.tensor(label).long(), dino_image
        else:
            return default_image, torch.tensor(label).long()

    def get_images_with_label(self, label):
        # Find indices of all images with the specified label
        indices = [idx for idx, lbl in enumerate(self.label_list) if lbl == label]
        
        # Retrieve the images with the specified label
        images = []
        for idx in indices:
            image_path = os.path.join(self.image_path, self.image_list[idx])
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        return images

fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']

path_dict = {
    # dataset_name: ["image_dir", "json_split_file"]
    "flower102": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["images", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["SUN397", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["101_ObjectCategories", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["UCF-101-midframes", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["2750", "data/data_splits/split_zhou_EuroSAT.json"]
}

def build_fewshot_dataset(set_id, root, transform, mode='train', n_shot=None, dino_transform=None):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, mode, n_shot, transform, dino_transform)
    path_suffix, json_path = path_dict[set_id.lower()]
    image_path = os.path.join(root, path_suffix)
    return BaseJsonDataset(image_path, json_path, mode, n_shot, transform, dino_transform)


class Aircraft(Dataset):
    """ FGVC Aircraft dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None, dino_transform=None):
        self.transform = transform
        self.dino_transform = dino_transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            default_image = self.transform(image)

        if self.dino_transform:
            dino_image = self.dino_transform(image)
            return default_image, torch.tensor(label).long(), dino_image
        else:
            return default_image, torch.tensor(label).long()

    def get_images_with_label(self, label):
        # Find indices of all images with the specified label
        indices = [idx for idx, lbl in enumerate(self.label_list) if lbl == label]
        
        # Retrieve the images with the specified label
        images = []
        for idx in indices:
            image_path = os.path.join(self.path, 'images', self.image_list[idx])
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        return images

