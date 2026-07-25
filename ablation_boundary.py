import random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image
from tqdm import tqdm
from medpy.metric.binary import hd95

# =========================
# 0) 配置
# =========================
DATA_ROOT = Path(r"D:\data set\Kvasir\Kvasir-SEG")
CKPT_PATH = Path(r"D:\projects\polypseg_baseline\runs\kvasir_unet_boundary_0.5\best.pth")
SAVE_ROOT = Path(r"D:\projects\polypseg_baseline\runs\ablation_boundary_final") # 图片保存位置
IMAGE_SIZE = 352
GLOBAL_SEED = 42
SPLIT_RATIO = 0.8
THRESH = 0.5 
MIN_AREA = 80 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 1) 模型结构
# =========================
class UnetMaskEdge(nn.Module):
    def __init__(self, encoder_name="resnet34"):
        super().__init__()
        base_unet = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.edge_head = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feats = self.encoder(x)
        dec_out = self.decoder(feats)
        logits_mask = self.segmentation_head(dec_out)
        logits_edge = self.edge_head(dec_out)
        return logits_mask, logits_edge

# =========================
# 2) 核心工具函数
# =========================
def get_metrics(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    h_val = 50.0
    if np.any(pred) and np.any(gt):
        try: h_val = hd95(pred, gt)
        except: h_val = 50.0
    elif not np.any(pred) and not np.any(gt):
        h_val = 0.0
    return dice, prec, rec, fp, h_val

# --- 后处理方法集 ---
def no_post(p, g, pe): return (p >= THRESH).astype(np.uint8)

def area_filter(p, g, pe):
    b = (p >= THRESH).astype(np.uint8)
    c, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(b)
    for cnt in c:
        if cv2.contourArea(cnt) >= MIN_AREA: cv2.drawContours(m, [cnt], -1, 1, cv2.FILLED)
    return m

def entropy_filter(p, g, pe):
    pr = np.clip(p, 1e-7, 1 - 1e-7)
    ent = - (pr * np.log2(pr) + (1 - pr) * np.log2(1 - pr))
    b = (p >= THRESH).astype(np.uint8)
    c, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(b)
    for cnt in c:
        mask_cnt = np.zeros_like(b)
        cv2.drawContours(mask_cnt, [cnt], -1, 1, cv2.FILLED)
        if np.mean(ent[mask_cnt == 1]) < 0.45: cv2.drawContours(m, [cnt], -1, 1, cv2.FILLED)
    return m

def prap_ours(p, g, pe):
    sm = (p >= 0.95).astype(np.uint8)
    k = np.ones((3,3), np.uint8)
    rm = cv2.dilate(sm, k, iterations=5) # 保持 iterations=5 
    mask_protected = (rm & (p >= THRESH)).astype(np.uint8)
    contours, _ = cv2.findContours(mask_protected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_m = np.zeros_like(p, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_AREA: cv2.drawContours(final_m, [cnt], -1, 1, cv2.FILLED)
    return final_m

# =========================
# 3) 主程序
# =========================
def main():
    # 确保保存目录存在
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    
    img_dir = Path(DATA_ROOT) / "images"
    stems = sorted([p.stem for p in img_dir.glob("*.jpg")])
    names = [s + ".jpg" for s in stems]
    random.seed(GLOBAL_SEED)
    random.shuffle(names)
    val_names = names[int(SPLIT_RATIO * len(names)):]

    class DS(Dataset):
        def __init__(self, n): self.n = n
        def __len__(self): return len(self.n)
        def __getitem__(self, i):
            nm = self.n[i]
            im = Image.open(DATA_ROOT / "images" / nm).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            ms = Image.open(DATA_ROOT / "masks" / nm).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
            return torch.from_numpy(np.array(im).astype(np.float32)/255).permute(2,0,1), torch.from_numpy((np.array(ms)>0).astype(np.uint8)), nm
    
    dl = DataLoader(DS(val_names), batch_size=1)
    model = UnetMaskEdge().to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()

    methods = {"1_No_Post": no_post, "2_Area": area_filter, "3_Entropy": entropy_filter, "4_PRAP_Ours": prap_ours}
    all_results = []

    print(f"📊 正在评估并生成可视化结果...")
    with torch.no_grad():
        for img, mask, name in tqdm(dl):
            logits_mask, logits_edge = model(img.to(DEVICE))
            prob = torch.sigmoid(logits_mask).squeeze().cpu().numpy()
            prob_edge = torch.sigmoid(logits_edge).squeeze().cpu().numpy()
            gt = mask.squeeze().numpy()
            
            pred_raw = no_post(prob, gt, prob_edge)
            _, _, _, fp_base, _ = get_metrics(pred_raw, gt)
            
            sample_info = {"name": name[0], "img": img, "gt": gt, "fp_base": fp_base, "m_metrics": {}}
            for m_n, func in methods.items():
                pred = func(prob, gt, prob_edge)
                d, p, r, fp, h = get_metrics(pred, gt)
                sample_info["m_metrics"][m_n] = {"pred": pred, "dice": d, "prec": p, "rec": r, "hd": h, "fp": fp}
            all_results.append(sample_info)

    # --- 排序并保存名单 ---
    all_results.sort(key=lambda x: x["fp_base"], reverse=True)
    vis_samples = all_results[:30] # 锁定前30个用于可视化
    with open("hard_samples_list.txt", "w") as f:
        for s in vis_samples: f.write(s["name"] + "\n")

    # --- 可视化图片保存逻辑 (这是为您新增的部分) ---
    print(f"🎨 正在导出可视化图像至: {SAVE_ROOT}")
    for s in vis_samples:
        img_bgr = cv2.cvtColor((s["img"].squeeze().permute(1,2,0).numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        c_gt, _ = cv2.findContours((s["gt"]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for m_n, m_data in s["m_metrics"].items():
            m_dir = SAVE_ROOT / m_n
            m_dir.mkdir(exist_ok=True)
            canvas = img_bgr.copy()
            c_pred, _ = cv2.findContours((m_data["pred"]*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, c_gt, -1, (0, 0, 255), 2)   # 红线: 真值
            cv2.drawContours(canvas, c_pred, -1, (0, 255, 0), 2) # 绿线: 预测
            cv2.imwrite(str(m_dir / f"{Path(s['name']).stem}.png"), canvas)

    # --- 汇总结果表 ---
    print("\n" + "="*100)
    print(f"{'Method':<18} | {'Dice ↑':<8} | {'Prec ↑':<8} | {'Rec ↑':<8} | {'HD95 ↓':<8} | {'Total FP'}")
    print("-" * 105)
    for m in methods:
        d_vals = [x["m_metrics"][m]["dice"] for x in all_results]; p_vals = [x["m_metrics"][m]["prec"] for x in all_results]
        r_vals = [x["m_metrics"][m]["rec"] for x in all_results]; h_vals = [x["m_metrics"][m]["hd"] for x in all_results]
        f_sum  = np.sum([x["m_metrics"][m]["fp"] for x in all_results])
        print(f"{m:<18} | {np.mean(d_vals):.4f} | {np.mean(p_vals):.4f} | {np.mean(r_vals):.4f} | {np.mean(h_vals):.4f} | {int(f_sum)}")
    print("="*105)

if __name__ == "__main__": main()