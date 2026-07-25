import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class KvasirSegDataset(Dataset):
    def __init__(self, root_dir: str, file_names, image_size=352, transform=None):
        self.root = Path(root_dir)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.file_names = file_names
        self.image_size = image_size
        self.transform = transform

        assert self.img_dir.exists(), f"Missing folder: {self.img_dir}"
        assert self.mask_dir.exists(), f"Missing folder: {self.mask_dir}"

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img_path = self.img_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.jpg"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # resize (simple + stable baseline)
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask)

        # binarize: 0/255 -> 0/1 (more robust: >0)
        mask = (mask > 0).astype(np.float32)

        if self.transform is not None:
            # Albumentations style: transform(image=..., mask=...)
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # to tensor: C,H,W
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()  # 1,H,W

        return image, mask

def list_stems(root_dir: str):
    img_dir = Path(root_dir) / "images"
    names = sorted([p.stem for p in img_dir.glob("*.jpg")])
    return names
