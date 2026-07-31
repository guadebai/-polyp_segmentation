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
EDGE_POS_WEIGHT = 30 # update here to run experiments on different edge_pos_weight values (1, 5, 10, 20, 30)
LAMBDA_EDGE = 0.1 # update here to run experiments on different lambda_edge values (0.1, 0.3, 0.5, 1)

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
BOUNDARY_OUT_DIR = Path("./runs/kvasir_unet_boundary" + "_lambda_" + str(LAMBDA_EDGE) + "_pos_" + str(EDGE_POS_WEIGHT))

# 2 Seed
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 3 Fixed data split
REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO_ROOT / "data_split" / "splits"

def list_stems(root_dir=DATA_ROOT):
    image_dir = Path(root_dir) / "images"
    mask_dir = Path(root_dir) / "masks"

    image_stems = {
        path.stem for path in image_dir.glob("*.jpg")
    }
    mask_stems = {
        path.stem for path in mask_dir.glob("*.jpg")
    }

    if image_stems != mask_stems:
        missing_masks = image_stems - mask_stems
        missing_images = mask_stems - image_stems

        raise ValueError(
            "Images and masks do not match.\n"
            f"Missing masks: {len(missing_masks)}\n"
            f"Missing images: {len(missing_images)}"
        )

    return sorted(image_stems)

def read_split_file(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}"
        )

    names = [
        line.strip()
        for line in file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(names) != len(set(names)):
        raise ValueError(
            f"Duplicate names found in: {file_path}"
        )

    return names

def get_data_split(root_dir=DATA_ROOT):
    # Read the split files to help persist the same split across runs
    train_names = read_split_file(
        SPLIT_DIR / "train.txt"
    )
    val_names = read_split_file(
        SPLIT_DIR / "val.txt"
    )
    test_names = read_split_file(
        SPLIT_DIR / "test.txt"
    )

    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)

    if not train_set.isdisjoint(val_set):
        raise ValueError("train and val overlap.")

    if not train_set.isdisjoint(test_set):
        raise ValueError("train and test overlap.")

    if not val_set.isdisjoint(test_set):
        raise ValueError("val and test overlap.")

    available_names = set(list_stems(root_dir))
    split_names = train_set | val_set | test_set

    missing_names = split_names - available_names
    extra_names = available_names - split_names

    if missing_names:
        raise ValueError(
            f"{len(missing_names)} split samples "
            "are missing from the dataset."
        )

    if extra_names:
        raise ValueError(
            f"{len(extra_names)} dataset samples "
            "are not included in the split files."
        )

    if len(train_names) != 700:
        raise ValueError(
            f"Expected 700 train samples, "
            f"found {len(train_names)}."
        )

    if len(val_names) != 150:
        raise ValueError(
            f"Expected 150 val samples, "
            f"found {len(val_names)}."
        )

    if len(test_names) != 150:
        raise ValueError(
            f"Expected 150 test samples, "
            f"found {len(test_names)}."
        )

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
        ],
        seed=SEED,
    )

    return A.Compose([], seed=SEED)


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