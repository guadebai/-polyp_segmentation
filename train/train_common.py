import random
from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch

# 1 Parameters
SEED = 42
DATA_ROOT = r"D:\data set\Kvasir\Kvasir-SEG"

IMAGE_SIZE = 352
EDGE_WIDTH = 3

BATCH_SIZE = 4
EPOCHS = 40
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

BCE_WEIGHT = 0.5
EDGE_POS_WEIGHT = 30.0
LAMBDA_EDGE = 1

CLIP_NORM = 0.25

SCHEDULER_STEP_SIZE = 10
SCHEDULER_GAMMA = 0.1

PATIENCE = 8
THRESHOLD = 0.5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EMPTY_HD95 = 50.0

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

BASELINE_OUT_DIR = Path("./runs/kvasir_unet_baseline")
BOUNDARY_OUT_DIR = Path("./runs/kvasir_unet_boundary_lambda_1")


# 2 Seed
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 3 Data split
def list_stems(root_dir=DATA_ROOT):
    image_dir = Path(root_dir) / "images"
    return sorted([path.stem for path in image_dir.glob("*.jpg")])


def get_data_split(root_dir=DATA_ROOT):
    names = list_stems(root_dir)

    rng = random.Random(SEED)
    rng.shuffle(names)

    total = len(names)

    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)

    train_names = names[:train_end]
    val_names = names[train_end:val_end]
    test_names = names[val_end:]

    return train_names, val_names, test_names


# 4 Transforms
def make_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.10,
                rotate_limit=10,
                p=0.3,
                border_mode=0
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.1
            ),
        ])

    return A.Compose([])


# 5 Model
class UnetBase(torch.nn.Module):
    def __init__(
        self,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS
    ):
        super().__init__()

        base_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )

        self.encoder = base_unet.encoder
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head

        self.decoder_out_channels = (
            self.segmentation_head[0].in_channels
        )

    def extract_decoder_features(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return decoder_output


class UnetBaseline(UnetBase):
    def forward(self, x):
        decoder_output = self.extract_decoder_features(x)
        logits_mask = self.segmentation_head(decoder_output)
        return logits_mask


class UnetBoundary(UnetBase):
    def __init__(
        self,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights
        )

        self.edge_head = torch.nn.Conv2d(
            self.decoder_out_channels,
            1,
            kernel_size=3,
            padding=1
        )

        torch.nn.init.xavier_uniform_(
            self.edge_head.weight,
            gain=1.0
        )
        torch.nn.init.constant_(
            self.edge_head.bias,
            0.0
        )

    def forward(self, x):
        decoder_output = self.extract_decoder_features(x)

        logits_mask = self.segmentation_head(decoder_output)
        logits_edge = self.edge_head(decoder_output)

        return logits_mask, logits_edge


# 6 Optimizer
def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


def make_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=SCHEDULER_STEP_SIZE,
        gamma=SCHEDULER_GAMMA
    )


# 7 HD95
def calculate_hd95(pred, gt):
    pred_exists = np.any(pred)
    gt_exists = np.any(gt)

    if pred_exists and gt_exists:
        from medpy.metric.binary import hd95
        return float(hd95(pred, gt))

    if not pred_exists and not gt_exists:
        return 0.0

    return EMPTY_HD95