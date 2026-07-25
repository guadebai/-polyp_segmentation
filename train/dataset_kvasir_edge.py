from pathlib import Path
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from PIL import Image


def list_stems(root_dir):
    image_dir = Path(root_dir) / "images"
    stems = sorted([p.stem for p in image_dir.glob("*.jpg")])
    return stems


def _mask_to_edge(mask01: np.ndarray, edge_width: int = 3) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.sum() == 0:
        return np.zeros_like(mask01)
    k = max(1, edge_width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask01, kernel, iterations=1)
    ero = cv2.erode(mask01, kernel, iterations=1)
    edge = (dil - ero) > 0
    return edge.astype(np.uint8)


class KvasirSegEdgeDataset(Dataset):
    """
    output: image, mask, edge
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

        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = (np.array(mask) >= 128).astype(np.uint8)

        # transform image and mask
        if self.transform is not None:
            aug = self.transform(
                image=image_np,
                mask=mask_np
            )
            image_np = aug["image"]
            mask_np = aug["mask"]

        mask_np = (np.asarray(mask_np) >= 0.5).astype(np.uint8)
        edge_np = _mask_to_edge(mask_np, edge_width=self.edge_width)

        # to tensor
        x = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1).float()
        y = torch.from_numpy(np.ascontiguousarray(mask_np)).unsqueeze(0).float()
        e = torch.from_numpy(np.ascontiguousarray(edge_np)).unsqueeze(0).float()

        return x, y, e, f"{stem}.jpg"
