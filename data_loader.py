from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from PIL import Image

import cfg


class CervicalDataset(Dataset):
    """Cervical dataset."""

    def __init__(self, data_path, patch_size, transform=None):
        self.data_path = data_path
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        roi, roi_label = self.load_data(idx)
        # img, label = self.crop_patch(roi, roi_label)
        # sample = {'img': img, 'label': label}
        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
        imgs, labels = self.full_crop_patch(roi, roi_label)
        for _, (img, label) in enumerate(zip(imgs, labels)):
            sample = {'img': img, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            imgs[_] = sample['img']
            labels[_] = sample['label']
        imgs = [_.unsqueeze(0) for _ in imgs]
        # labels = [_.unsqueeze(0) for _ in labels]
        imgs = torch.cat(imgs, 0)
        # labels = torch.cat(labels, 0)
        sample = {'img': imgs, 'label': labels}

        return sample

    def full_crop_patch(self, img, labels):
        H, W, _ = img.shape
        px, py = self.patch_size
        obj_num = labels.shape[0]
        if obj_num > 30:
            idx = np.random.permutation(obj_num)
            labels = labels[idx[:30]]
        cropPatch, cropLabel = [], []

        for label in labels:
            ux, uy, vx, vy = label
            nx = np.random.randint(max(vx - px, 0), min(ux + 1, W - px + 1))
            ny = np.random.randint(max(vy - py, 0), min(uy + 1, H - py + 1))

            patch_coord = np.zeros((1, 4), dtype="int")
            patch_coord[0, 0] = nx
            patch_coord[0, 1] = ny
            patch_coord[0, 2] = nx + px
            patch_coord[0, 3] = ny + py

            patch = img[ny: ny + py, nx: nx + px, :]

            cropPatch.append(patch)

            index = self.compute_overlap(patch_coord, labels, 0.5)
            index = np.squeeze(index, axis=0)
            label = labels[index, :]
            label[:, 0] = np.maximum(label[:, 0] - nx, 0)
            label[:, 1] = np.maximum(label[:, 1] - ny, 0)
            label[:, 2] = np.minimum(label[:, 2] - nx, px)
            label[:, 3] = np.minimum(label[:, 3] - ny, py)

            cropLabel.append(label)

        return cropPatch, cropLabel

    def crop_patch(self, img, labels):
        H, W, _ = img.shape
        px, py = self.patch_size
        obj_num = labels.shape[0]
        index = np.random.randint(0, obj_num)
        ux, uy, vx, vy = labels[index, :]

        nx = np.random.randint(max(vx - px, 0), min(ux + 1, W - px + 1))
        ny = np.random.randint(max(vy - py, 0), min(uy + 1, H - py + 1))

        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py

        index = self.compute_overlap(patch_coord, labels, 0.5)
        index = np.squeeze(index, axis=0)

        patch = img[ny: ny + py, nx: nx + px, :]
        label = labels[index, :]
        label[:, 0] = np.maximum(label[:, 0] - patch_coord[0, 0], 0)
        label[:, 1] = np.maximum(label[:, 1] - patch_coord[0, 1], 0)
        label[:, 2] = np.minimum(cfg.patch_size[0] + label[:, 2] - patch_coord[0, 2], cfg.patch_size[0])
        label[:, 3] = np.minimum(cfg.patch_size[1] + label[:, 3] - patch_coord[0, 3], cfg.patch_size[1])
        return patch, label

    def compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ---------- calculate iou
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index

    def load_data(self, index):
        data = np.load(self.data_path[index])
        img, label = data['img'], data['label']
        img = img.astype(np.float32) / 255.0

        return img, label


class WsiDataset(Dataset):
    def __init__(self, read, y_num, x_num, strides, coordinates, patch_size, transform=None):
        self.read = read
        self.y_num = y_num
        self.x_num = x_num
        self.strieds = strides
        self.coordinates = coordinates
        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        coord_y, coord_x = self.coordinates[index]
        img = self.read.ReadRoi(coord_x, coord_y, cfg.patch_size[0], cfg.patch_size[1], scale=20).copy()

        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255.0

        return torch.from_numpy(img).float(), coord_y, coord_x

    def __len__(self):
        return self.y_num * self.x_num


def full_collater(data):
    imgs = [s['img'] for s in data] # 1, bz, W H C
    labels = [s['label'] for s in data] # 1, bz, n, 4

    imgs = torch.stack(imgs, dim=0).squeeze(0) # bz, W H C
    imgs = imgs.permute((0, 3, 1, 2))
    labels = labels[0]

    max_num_labels = max(label.shape[0] for label in labels)

    if max_num_labels > 0:
        label_pad = torch.ones((len(labels), max_num_labels, 4)) * -1

        for idx, label in enumerate(labels):
            if label.shape[0] > 0:
                label_pad[idx, :label.shape[0], :] = label
    else:
        label_pad = torch.ones((len(labels), 1, 4)) * -1

    return {'img': imgs, 'label': label_pad}

def collater(data):
    imgs = [s['img'] for s in data]
    labels = [s['label'] for s in data]

    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.permute((0, 3, 1, 2))

    max_num_labels = max(label.shape[0] for label in labels)

    if max_num_labels > 0:
        label_pad = torch.ones((len(labels), max_num_labels, 4)) * -1

        for idx, label in enumerate(labels):
            if label.shape[0] > 0:
                label_pad[idx, :label.shape[0], :] = label
    else:
        label_pad = torch.ones((len(labels), 1, 4)) * -1

    return {'img': imgs, 'label': label_pad}
