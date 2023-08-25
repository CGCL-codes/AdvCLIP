import os
import pickle
import h5py
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.utils.data as data
from torchvision import models, transforms
from torch.utils.data import DataLoader


class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labs,
            ids
    ):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.ids = ids

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        id = self.ids[index]
        return img, text, lab, id

    def __len__(self):
        count = len(self.texts)
        return count

def load_dataset(name, bsz):
    train_loc = 'data/' + name + '/train.pkl'
    test_loc = 'data/' + name + '/test.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        train_texts = data['text']
        train_images = data['image']
        train_ids = data['ids']
    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']
        test_texts = data['text']
        test_images = data['image']
        test_ids = data['ids']
    imgs = {'train': train_images, 'test': test_images}
    texts = {'train': train_texts, 'test': test_texts}
    labs = {'train': train_labels, 'test': test_labels}
    ids = {'train': train_ids, 'test': test_ids}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x], ids=ids[x])
               for x in ['train', 'test']}

    shuffle = {'train': True, 'test': False}

    dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}

    return dataloader