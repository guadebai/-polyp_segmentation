import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm

# =========================
# 0) Paths & Params
# =========================
DATA_ROOT = Path(r"D:\data set\Kvasir\Kvasir-SEG")
CKPT_PATH = Path(r"D:\projects\polypseg_baseline\runs\kvasir_unet_baseline\best.pth")

IMAGE_SIZE = 352
THRESH = 0.3                 # ★ 用你 detection 最优的阈值
MAX_SAVE = 30                # 保存多少张论文示例图

OUT_DIR = Path(
    fr"D:\projects\polypseg_baseline\runs\contour_vis_thr{THRESH}"
)

SPLIT_SEED = 42
SPLIT_RATIO = 0.8


# =========================
# 1) Dataset
# =========================
class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, names, image_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.names = names
        self.image_size = image_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        image = Image.open(self.image_dir / name).convert("RGB")
        mask = Image.open(self.mask_dir / name).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        img_np = np.array(image).astype(np.float32) / 255.0
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        x = torch.from_numpy(img_np).permute(2, 0, 1)
        y = torch.from_numpy(mask_np)

        return x, y, name


# =========================
# 2) Utils
# =========================
def mask_to_contours(mask01):
    """
    mask01: (H,W) uint8 0/1
    return: contours for cv2.drawContours
    """
    mask255 = (mask01 * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def make_split(image_dir):
    stems = sorted([p.stem for p in image_dir.glob("*.jpg")])
    names = [s + ".jpg" for s in stems]

    random.shuffle(names)
    split = int(SPLIT_RATIO * len(names))
    return names[:split], names[split:]


# =========================
# 3) Main
# =========================
def main():
    random.seed(SPLIT_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    image_dir = DATA_ROOT / "images"
    mask_dir = DATA_ROOT / "masks"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, val_names = make_split(image_dir)
    ds = KvasirSegDataset(image_dir, mask_dir, val_names, IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    # 核心修改：过滤权重，兼容边缘监督模型（跳过edge_head）
    state = torch.load(CKPT_PATH, map_location=device)
    state = {k: v for k, v in state.items() if k in model.state_dict()}  # 只保留模型有的键
    model.load_state_dict(state, strict=False)  # 忽略不匹配的权重（如edge_head）
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    print(f"Loaded model: {CKPT_PATH}")
    print(f"Saving contour figures to: {OUT_DIR}")

    saved = 0

    with torch.no_grad():
        for x, gt_mask, name in tqdm(dl, desc="Draw contours"):
            if saved >= MAX_SAVE:
                break

            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (prob >= THRESH).astype(np.uint8)

            gt_mask = gt_mask[0].numpy()

            img_rgb = (x[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # contours
            gt_contours = mask_to_contours(gt_mask)
            pred_contours = mask_to_contours(pred_mask)

            # draw: GT=red, Pred=green
            cv2.drawContours(img_bgr, gt_contours, -1, (0, 0, 255), 2)
            cv2.drawContours(img_bgr, pred_contours, -1, (0, 255, 0), 2)

            out_path = OUT_DIR / f"{Path(name[0]).stem}_contour.png"
            cv2.imwrite(str(out_path), img_bgr)

            saved += 1

    print(f"Done. Saved {saved} contour figures.")


if __name__ == "__main__":
    main()
