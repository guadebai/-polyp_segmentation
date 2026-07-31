from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from train.train_common import DATA_ROOT, get_data_split

def save_names(path: Path, names: list[str]) -> None:
    path.write_text(
        "\n".join(names) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    train_names, val_names, test_names = get_data_split(DATA_ROOT)

    # check no overlap
    train_set = set(train_names)
    val_set = set(val_names)
    test_set = set(test_names)

    assert train_set.isdisjoint(val_set), "found overlap between train and val"
    assert train_set.isdisjoint(test_set), "foun overlap between train and test"
    assert val_set.isdisjoint(test_set), "found overlap between val and test"

    total = len(train_names) + len(val_names) + len(test_names)

    assert len(train_names) == 700, f"incorrect train count：{len(train_names)}"
    assert len(val_names) == 150, f"incorrect val count：{len(val_names)}"
    assert len(test_names) == 150, f"incorrect test count：{len(test_names)}"
    assert total == 1000, f"incorrect total count：{total}"

    split_dir = Path(__file__).resolve().parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    save_names(split_dir / "train.txt", train_names)
    save_names(split_dir / "val.txt", val_names)
    save_names(split_dir / "test.txt", test_names)

    print("Data split lists saved successfully:")
    print(f"train: {len(train_names)} -> {split_dir / 'train.txt'}")
    print(f"val:   {len(val_names)} -> {split_dir / 'val.txt'}")
    print(f"test:  {len(test_names)} -> {split_dir / 'test.txt'}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()