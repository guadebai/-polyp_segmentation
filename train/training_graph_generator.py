from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(
    r"D:\projects\polypseg\train\history.csv"
)
OUTPUT_DIR = CSV_PATH.parent

BEST_EPOCH = 23
EARLY_STOP_EPOCH = 31


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # 兼容两种列名
    if "val_soft_dice" in df.columns:
        dice_column = "val_soft_dice"
    elif "val_dice" in df.columns:
        dice_column = "val_dice"
    else:
        raise ValueError(
            "CSV must contain either 'val_soft_dice' or 'val_dice'. "
            f"Current columns: {list(df.columns)}"
        )

    required_columns = {"epoch", "train_loss", dice_column}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in CSV: {sorted(missing)}")

    best_rows = df.loc[df["epoch"] == BEST_EPOCH]

    if best_rows.empty:
        raise ValueError(f"Epoch {BEST_EPOCH} was not found in the CSV.")

    best_dice = best_rows[dice_column].iloc[0]

    # Figure 3(a): training loss
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        df["epoch"],
        df["train_loss"],
        linewidth=1.8,
    )
    plt.axvline(
        BEST_EPOCH,
        linestyle="--",
        linewidth=1.2,
        label=f"Best checkpoint: epoch {BEST_EPOCH}",
    )
    plt.axvline(
        EARLY_STOP_EPOCH,
        linestyle=":",
        linewidth=1.2,
        label=f"Early stopping: epoch {EARLY_STOP_EPOCH}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_loss_curve.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    # Figure 3(b): validation soft Dice
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        df["epoch"],
        df[dice_column],
        linewidth=1.8,
    )
    plt.scatter(
        [BEST_EPOCH],
        [best_dice],
        s=35,
        label=f"Best soft Dice: {best_dice:.4f}",
        zorder=3,
    )
    plt.axvline(
        EARLY_STOP_EPOCH,
        linestyle=":",
        linewidth=1.2,
        label=f"Early stopping: epoch {EARLY_STOP_EPOCH}",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation soft Dice")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "validation_soft_dice_curve.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    print("Figures saved to:")
    print(OUTPUT_DIR / "training_loss_curve.png")
    print(OUTPUT_DIR / "validation_soft_dice_curve.png")


if __name__ == "__main__":
    main()