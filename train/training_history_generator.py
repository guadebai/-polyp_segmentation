import re
from pathlib import Path

import pandas as pd


LOG_PATH = Path(r"D:\projects\polypseg\train\train_log.txt")
CSV_PATH = LOG_PATH.with_name("history.csv")

pattern = re.compile(
    r"Epoch\s+(\d+)(?:/\d+)?.*?"
    r"train[_ ]?loss[=:]\s*([0-9.]+).*?"
    r"val(?:idation)?[_ ]?(?:soft[_ ]?)?dice[=:]\s*([0-9.]+)",
    re.IGNORECASE,
)


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Training log not found: {LOG_PATH}")

    records = []

    for line in LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            records.append(
                {
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_soft_dice": float(match.group(3)),
                }
            )

    if not records:
        raise ValueError(
            "No epoch records were found. Check whether the log format matches the regex."
        )

    df = pd.DataFrame(records).drop_duplicates(subset="epoch").sort_values("epoch")
    df.to_csv(CSV_PATH, index=False)

    print(df)
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    main()