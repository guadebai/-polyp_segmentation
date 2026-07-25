import random
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from medpy.metric.binary import hd95 

# =========================
# 0) 配置
# =========================
DATA_ROOT = Path(r"D:\data set\Kvasir\Kvasir-SEG")
CKPT_PATH = Path(r"D:\projects\polypseg_baseline\runs\kvasir_unet_baseline\best.pth")
HARD_LIST_PATH = Path("hard_samples_list.txt") 
IMAGE_SIZE = 352
SPLIT_RATIO = 0.8
THRESH = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GLOBAL_SEED = 42

# =========================
# 1) 数据集逻辑 (保持 Seed 一致)
# =========================
def get_val_split(root_dir, seed=42):
    image_dir = Path(root_dir) / "images"
    stems = sorted([p.stem for p in image_dir.glob("*.jpg")])
    names = [s + ".jpg" for s in stems]
    random.seed(seed)
    random.shuffle(names)
    split = int(SPLIT_RATIO * len(names))
    return names[split:]

class KvasirSegDataset(Dataset):
    def __init__(self, root_dir, names, size=352):
        self.root_dir = Path(root_dir)
        self.names = names
        self.size = size
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(self.root_dir / "images" / name).convert('RGB').resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(self.root_dir / "masks" / name).convert('L').resize((self.size, self.size), Image.NEAREST)
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        return torch.from_numpy(img_np).permute(2, 0, 1), torch.from_numpy(mask_np), name

# =========================
# 2) 核心指标函数 (补全 Precision/Recall)
# =========================
def get_metrics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    prec = tp / (tp + fp + 1e-7)  # 精确率：查准率
    rec = tp / (tp + fn + 1e-7)   # 召回率：查全率
    
    h_val = 50.0
    if np.any(pred) and np.any(gt):
        try: h_val = hd95(pred, gt)
        except: h_val = 50.0
    elif not np.any(pred) and not np.any(gt):
        h_val = 0.0
        
    return dice, prec, rec, h_val, fp

# =========================
# 3) 执行评估
# =========================
def main():
    all_val_names = get_val_split(DATA_ROOT, seed=GLOBAL_SEED)
    hard_names = []
    if HARD_LIST_PATH.exists():
        with open(HARD_LIST_PATH, "r") as f:
            hard_names = [line.strip() for line in f.readlines() if line.strip()]

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    ds = KvasirSegDataset(DATA_ROOT, all_val_names, size=IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=1)
    
    # 结果容器
    global_res = {"dice":[], "prec":[], "rec":[], "hd":[], "fp":[]}
    hard_res = {"dice":[], "prec":[], "rec":[], "hd":[], "fp":[]}

    print(f"🚀 正在对验证集进行全指标评估...")
    with torch.no_grad():
        for img, mask, name in tqdm(dl):
            out = model(img.to(DEVICE))
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            gt = mask.squeeze().numpy()
            pred = (prob >= THRESH).astype(np.uint8)
            
            d, p, r, h, fp = get_metrics(pred, gt)
            
            for k, v in zip(global_res.keys(), [d, p, r, h, fp]):
                global_res[k].append(v)
            
            if name[0] in hard_names:
                for k, v in zip(hard_res.keys(), [d, p, r, h, fp]):
                    hard_res[k].append(v)

    # --- 最终专业输出表 ---
    print("\n" + "="*85)
    print(f"{'Metric Type':<18} | {'Dice ↑':<7} | {'Prec ↑':<7} | {'Rec ↑':<7} | {'HD95 ↓':<7} | {'Total FP'}")
    print("-" * 85)
    print(f"{'GLOBAL (All 200)':<18} | {np.mean(global_res['dice']):.4f} | {np.mean(global_res['prec']):.4f} | {np.mean(global_res['rec']):.4f} | {np.mean(global_res['hd']):.4f} | {int(np.sum(global_res['fp']))}")
    if hard_names:
        print(f"{'HARD (Top 30)':<18} | {np.mean(hard_res['dice']):.4f} | {np.mean(hard_res['prec']):.4f} | {np.mean(hard_res['rec']):.4f} | {np.mean(hard_res['hd']):.4f} | {int(np.sum(hard_res['fp']))}")
    print("="*85)

if __name__ == "__main__":
    main()