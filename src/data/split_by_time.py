from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_behavior.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

TRAIN_END_DATE = "2017-12-01"
VALIDATION_DATE = "2017-12-02"
TEST_DATE = "2017-12-03"


def print_summary(name: str, df: pd.DataFrame) -> None:
    """Print summary statistics for a data split."""

    purchase_count = (df["behavior_type"] == "buy").sum()

    print(f"\n{name}")
    print(f"Rows:         {len(df):,}")
    print(f"Unique users: {df['user_id'].nunique():,}")
    print(f"Unique items: {df['item_id'].nunique():,}")
    print(f"Purchases:    {purchase_count:,}")
    print(
        f"Date range:   {df['event_date'].min()} "
        f"to {df['event_date'].max()}"
    )


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns exist."""

    required_columns = {
        "user_id",
        "item_id",
        "category_id",
        "behavior_type",
        "timestamp",
        "event_time",
        "event_date",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    print(f"Loading cleaned data from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)

    validate_columns(df)

    print(f"Loaded rows: {len(df):,}")

    df["event_date"] = df["event_date"].astype("string")

    train = df[
        df["event_date"] <= TRAIN_END_DATE
    ].copy()

    validation = df[
        df["event_date"] == VALIDATION_DATE
    ].copy()

    test = df[
        df["event_date"] == TEST_DATE
    ].copy()

    if train.empty:
        raise ValueError("Training set is empty.")

    if validation.empty:
        raise ValueError("Validation set is empty.")

    if test.empty:
        raise ValueError("Test set is empty.")

    total_split_rows = len(train) + len(validation) + len(test)

    if total_split_rows != len(df):
        raise ValueError(
            "Split row count does not match cleaned data: "
            f"{total_split_rows:,} != {len(df):,}"
        )

    train_dates = set(train["event_date"])
    validation_dates = set(validation["event_date"])
    test_dates = set(test["event_date"])

    if train_dates & validation_dates:
        raise ValueError(
            "Training and validation date ranges overlap."
        )

    if train_dates & test_dates:
        raise ValueError(
            "Training and test date ranges overlap."
        )

    if validation_dates & test_dates:
        raise ValueError(
            "Validation and test date ranges overlap."
        )

    sort_columns = ["user_id", "event_time", "item_id"]

    train = train.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    validation = validation.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    test = test.sort_values(
        sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.parquet"
    validation_path = OUTPUT_DIR / "validation.parquet"
    test_path = OUTPUT_DIR / "test.parquet"

    train.to_parquet(
        train_path,
        index=False,
        compression="snappy",
    )

    validation.to_parquet(
        validation_path,
        index=False,
        compression="snappy",
    )

    test.to_parquet(
        test_path,
        index=False,
        compression="snappy",
    )

    print_summary("Training Set", train)
    print_summary("Validation Set", validation)
    print_summary("Test Set", test)

    print(f"\nTotal split rows: {total_split_rows:,}")
    print("\nSaved files:")
    print(f"  Training:   {train_path}")
    print(f"  Validation: {validation_path}")
    print(f"  Test:       {test_path}")


if __name__ == "__main__":
    main()