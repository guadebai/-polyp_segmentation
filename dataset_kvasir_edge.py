import os
from pathlib import Path
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset
from PIL import Image


def list_stems(root_dir):
    """
    返回 images 目录下所有文件的 stem（不带扩展名），保持和你 baseline 一致
    """
    image_dir = Path(root_dir) / "images"
    stems = sorted([p.stem for p in image_dir.glob("*.jpg")])
    return stems


def _mask_to_edge(mask01: np.ndarray, edge_width: int = 3) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.sum() == 0:  # 空mask，直接返回全0边缘
        return np.zeros_like(mask01)
    k = max(1, edge_width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask01, kernel, iterations=1)
    ero = cv2.erode(mask01, kernel, iterations=1)
    edge = (dil - ero) > 0
    return edge.astype(np.uint8)


class KvasirSegEdgeDataset(Dataset):
    """
    输出：image, mask, edge
    image: float tensor (3,H,W) in [0,1]
    mask : float tensor (1,H,W) 0/1
    edge : float tensor (1,H,W) 0/1
    """
    def __init__(self,
                 root_dir,
                 file_names,
                 image_size=352,
                 transform=None,
                 edge_width=3):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "images"
        self.mask_dir = self.root_dir / "masks"
        self.file_names = file_names  # stems list
        self.image_size = int(image_size)
        self.transform = transform
        self.edge_width = int(edge_width)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        stem = self.file_names[idx]
        img_path = self.image_dir / f"{stem}.jpg"
        msk_path = self.mask_dir / f"{stem}.jpg"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(msk_path).convert("L")

        # resize：和你 baseline 一致（训练时 transform 里可能会做，但我们这里保证一致性）
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0           # (H,W,3)
        mask_np = (np.array(mask) > 0).astype(np.uint8)                 # (H,W) 0/1
        edge_np = _mask_to_edge(mask_np, edge_width=self.edge_width)    # (H,W) 0/1

        # albumentations：统一对 image/mask/edge 做同样变换（关键！）
        if self.transform is not None:
            aug = self.transform(image=image_np, masks=[mask_np, edge_np])
            image_np = aug["image"]
            mask_np, edge_np = aug["masks"]

        # to tensor
        x = torch.from_numpy(image_np).permute(2, 0, 1).float()         # (3,H,W)
        y = torch.from_numpy(mask_np).unsqueeze(0).float()              # (1,H,W)
        e = torch.from_numpy(edge_np).unsqueeze(0).float()              # (1,H,W)

        return x, y, e, f"{stem}.jpg"
