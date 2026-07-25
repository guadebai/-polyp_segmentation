# evaluate_detection.py
import random
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm


# 0 Paths & Params
DATA_ROOT = Path(r"D:\data set\Kvasir\Kvasir-SEG") 

CKPT_PATH = Path(r"D:\projects\polypseg_baseline\runs\kvasir_unet_baseline\best.pth") 

IMAGE_SIZE = 352

# detection parameters
THRESH = 0.8
MIN_AREA = 100 
IOU_THR = 0.5 

# split method, same as train_unet.py（sorted + random.shuffle + 80/20）
SPLIT_SEED = 42
SPLIT_RATIO = 0.8  # 80/20

# blue box GT，green box Pred
SAVE_VIS = True
VIS_MAX = 50

VIS_DIR = Path(rf"D:\projects\polypseg_baseline\runs\detection_vis")

# 1 Dataset: read image + mask
class KvasirSegDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, names, image_size: int):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.names = names
        self.image_size = image_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = self.image_dir / name
        mask_path = self.mask_dir / name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # resize, same as traing
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0  # (H,W,3) -> [0,1]
        mask_np = np.array(mask)
        mask_np = (mask_np > 0).astype(np.uint8)               # (H,W) -> 0/1

        # to tensor
        x = torch.from_numpy(image_np).permute(2, 0, 1)        # (3,H,W)
        y = torch.from_numpy(mask_np).unsqueeze(0).float()     # (1,H,W)

        return x, y, name


# 2 mask -> bboxes (connection)
def mask_to_bboxes(mask01: np.ndarray, min_area=50):
    """
    mask01: (H,W) uint8 0/1
    return: list of [x1,y1,x2,y2]
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])
    return bboxes


def compute_iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def detection_metrics_one_image(gt_boxes, pred_boxes, iou_thr=0.5):
    """
    TP/FP/FN
    """
    matched = set()
    tp = 0

    for pb in pred_boxes:
        best_i, best_iou = -1, 0.0
        for i, gb in enumerate(gt_boxes):
            if i in matched:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_iou >= iou_thr and best_i >= 0:
            tp += 1
            matched.add(best_i)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def draw_bboxes(image_uint8_bgr, gt_boxes, pred_boxes):
    """
    GT = blue box, Pred = green box
    """
    img = image_uint8_bgr.copy()

    # GT: blue
    for x1, y1, x2, y2 in gt_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Pred: green
    for x1, y1, x2, y2 in pred_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img


# 3 Split：same as train_unet.py
def make_split(image_dir: Path):
    names = sorted([p.name for p in image_dir.glob("*.jpg")])

    random.shuffle(names)
    split = int(SPLIT_RATIO * len(names))
    train_names = names[:split]
    val_names = names[split:]
    return train_names, val_names


# 4 Main
def main():
    random.seed(SPLIT_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    image_dir = DATA_ROOT / "images"
    mask_dir = DATA_ROOT / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Couldn't find images/masks folder: {image_dir} / {mask_dir}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"best.pth does not exist：{CKPT_PATH}")

    _, val_names = make_split(image_dir)
    print(f"Val images: {len(val_names)} (split=80/20, seed={SPLIT_SEED})")
    print("VAL first5:", val_names[:5])

    ds_val = KvasirSegDataset(image_dir, mask_dir, val_names, IMAGE_SIZE)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)

    # same as train (resnet34 + Unet + 1 class)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    state = torch.load(CKPT_PATH, map_location=device)
    state = {k: v for k, v in state.items() if k in model.state_dict()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    VIS_MAX = 50 
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_VIS:
        print(f"Save detection visualizations to: {VIS_DIR} (max {VIS_MAX})")
    else:
        print(f"Metrics will be saved to: {VIS_DIR}")

    TP = FP = FN = 0
    vis_count = 0
   
    with torch.no_grad():
        for x, y, name in tqdm(dl_val, desc="Detect(val)"):
            x = x.to(device)  # (1,3,H,W)
            y = y.to(device)  # (1,1,H,W)

            logits = model(x)  # (1,1,H,W)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred01 = (prob >= THRESH).astype(np.uint8)

            gt01 = (y[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

            gt_boxes = mask_to_bboxes(gt01, min_area=MIN_AREA)
            pred_boxes = mask_to_bboxes(pred01, min_area=MIN_AREA)

            tp, fp, fn = detection_metrics_one_image(gt_boxes, pred_boxes, IOU_THR)
            TP += tp
            FP += fp
            FN += fn

            if SAVE_VIS and vis_count < VIS_MAX:
                img_rgb = (x[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                vis = draw_bboxes(img_bgr, gt_boxes, pred_boxes)

                out_path = VIS_DIR / f"{Path(name[0]).stem}_det.png"
                cv2.imwrite(str(out_path), vis)
                vis_count += 1

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n=== Detection metrics (derived from segmentation masks) ===")
    print(f"TP={TP}  FP={FP}  FN={FN}")
    print(f"Precision={precision:.4f}")
    print(f"Recall/Sensitivity={recall:.4f}")
    print(f"F1={f1:.4f}")
    print(f"IoU threshold={IOU_THR}, prob thresh={THRESH}, min_area={MIN_AREA}")
    print(f"Saved to: {VIS_DIR}")

    out_txt = VIS_DIR / "metrics.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Detection metrics (derived from segmentation masks) ===\n")
        f.write(f"TP={TP}  FP={FP}  FN={FN}\n")
        f.write(f"Precision={precision:.4f}\n")
        f.write(f"Recall/Sensitivity={recall:.4f}\n")
        f.write(f"F1={f1:.4f}\n")
        f.write(f"IoU threshold={IOU_THR}, prob thresh={THRESH}, min_area={MIN_AREA}\n")
        f.write(f"split_ratio={SPLIT_RATIO}, seed={SPLIT_SEED}\n")
        f.write(f"checkpoint={CKPT_PATH}\n")

    print(f"metrics.txt saved: {out_txt}")


if __name__ == "__main__":
    main()
