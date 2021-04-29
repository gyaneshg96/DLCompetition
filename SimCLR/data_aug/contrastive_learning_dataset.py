from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

import torch
import os
import natsort
from PIL import Image
import numpy as np 
import math
import logging
from torch.utils.data import Dataset

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform, labels = "NA"):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.labels = labels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform is not None:
          image = self.transform(image)
        if self.labels == "NA":
          return (image, torch.Tensor([-1])) # label should be unused for this set
        else:
          return (image, self.labels[idx].squeeze())

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1, num_aug=5):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        if num_aug == 5:
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.8),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlur(kernel_size=int(0.1 * size)),
                                                  transforms.ToTensor()])
        elif num_aug == 7:
            data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomApply([color_jitter], p=0.8),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  GaussianBlur(kernel_size=int(0.1 * size)),
                                                  transforms.RandomRotation(degrees=45),
                                                  transforms.RandomAffine(degrees=45),
                                                  transforms.ToTensor()])
        return data_transforms

    
    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True), 
                          'dldataset': lambda: CustomDataSet(self.root_folder + '/unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(128),
                                                              n_views)),
                          'dldataset_aug7': lambda: CustomDataSet(self.root_folder + '/unlabeled',
                                                             transform=ContrastiveLearningViewGenerator(
                                                                 self.get_simclr_pipeline_transform(96, num_aug=7),
                                                                 n_views))
                          }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
