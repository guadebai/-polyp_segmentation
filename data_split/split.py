import random
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from train.train_common import (
    DATA_ROOT,
    SEED,
    TRAIN_RATIO,
    VAL_RATIO,
)


def save_names(path: Path, names: list[str]) -> None:
    path.write_text(
        "\n".join(names) + "\n",
        encoding="utf-8",
    )


def list_stems(root_dir: str) -> list[str]:
    image_dir = Path(root_dir) / "images"
    mask_dir = Path(root_dir) / "masks"

    image_stems = {
        path.stem for path in image_dir.glob("*.jpg")
    }
    mask_stems = {
        path.stem for path in mask_dir.glob("*.jpg")
    }

    if image_stems != mask_stems:
        raise ValueError("Images and masks do not match.")

    return sorted(image_stems)


def main() -> None:
    names = list_stems(DATA_ROOT)

    rng = random.Random(SEED)
    rng.shuffle(names)

    total = len(names)
    train_end = int(TRAIN_RATIO * total)
    val_end = train_end + int(VAL_RATIO * total)

    train_names = names[:train_end]
    val_names = names[train_end:val_end]
    test_names = names[val_end:]

    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)

    if not train_set.isdisjoint(val_set):
        raise ValueError("Found overlap between train and val.")

    if not train_set.isdisjoint(test_set):
        raise ValueError("Found overlap between train and test.")

    if not val_set.isdisjoint(test_set):
        raise ValueError("Found overlap between val and test.")

    if len(train_names) != 700:
        raise ValueError(
            f"Incorrect train count: {len(train_names)}"
        )

    if len(val_names) != 150:
        raise ValueError(
            f"Incorrect val count: {len(val_names)}"
        )

    if len(test_names) != 150:
        raise ValueError(
            f"Incorrect test count: {len(test_names)}"
        )

    split_dir = Path(__file__).resolve().parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    save_names(split_dir / "train.txt", train_names)
    save_names(split_dir / "val.txt", val_names)
    save_names(split_dir / "test.txt", test_names)

    print("Data split lists generated successfully:")
    print(f"train: {len(train_names)}")
    print(f"val:   {len(val_names)}")
    print(f"test:  {len(test_names)}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()