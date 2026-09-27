from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "user_behavior_sample.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_behavior.parquet"
)

DAILY_STATS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
    / "daily_behavior_counts"
)

VALID_BEHAVIOR = ("pv", "fav", "cart", "buy")

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Can't find sampled data:{INPUT_PATH}")

    df = pd.read_parquet(INPUT_PATH)

    original_rows = len(df)
    print(f"Original row number: {original_rows:,}")
    
    required_columns = [
        "user_id",
        "item_id",
        "category_id",
        "behavior_type",
        "timestamp"
    ]
    df = df.dropna(subset=required_columns)
    after_null_rows = len(df)

    df = df.drop_duplicates(subset=required_columns)
    after_duplicate_rows = len(df)

    df["behavior_type"] = df["behavior_type"].astype("string")
    df = df[df["behavior_type"].isin(VALID_BEHAVIOR)]
    after_behavior_rows = len(df)

    df = df[
        (df["user_id"] > 0)
        & (df["item_id"] > 0)
        & (df["category_id"] > 0)
        & (df["timestamp"] > 0)
    ]
    df["event_time"] = (
        pd.to_datetime(
            df["timestamp"],
            unit="s",
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert("Asia/Shanghai")
    )
    df = df.dropna(subset=["event_time"])

    start_time = pd.Timestamp(
    "2017-11-25 00:00:00",
    tz="Asia/Shanghai",
    )
    end_time = pd.Timestamp(
        "2017-12-04 00:00:00",
        tz="Asia/Shanghai",
    )

    before_time_filter = len(df)

    df = df[
        (df["event_time"] >= start_time)
        & (df["event_time"] < end_time)
    ].copy()

    print(
        f"Number of records out of event period:"
        f"{before_time_filter - len(df):,}"
    )

    df["event_date"] = df["event_time"].dt.strftime("%Y-%m-%d")
    
    df = df.sort_values(
        ["user_id", "event_time", "item_id"],
        kind="stable",
    ).reset_index(drop=True)

    cleaned_rows = len(df)

    daily_stats = (
        df.groupby(["event_date", "behavior_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    daily_stats = daily_stats.reindex(
        columns=["pv", "fav", "cart", "buy"],
        fill_value=0,
    )
    daily_stats["total"] = daily_stats.sum(axis=1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DAILY_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )
    daily_stats.to_csv(
        DAILY_STATS_PATH,
        encoding="utf-8-sig",
    )

    print("\nCleaning Summary")
    print(f"Original rows: {original_rows:,}")
    print(f"Rows after removing nulls: {after_null_rows:,}")
    print(f"Rows after deduplication: {after_duplicate_rows:,}")
    print(f"Rows after behavior filter: {after_behavior_rows:,}")
    print(f"Final retained rows: {cleaned_rows:,}")
    print(f"Total removed rows: {original_rows - cleaned_rows:,}")

    print(f"\nUnique users: {df['user_id'].nunique():,}")
    print(f"Unique items: {df['item_id'].nunique():,}")
    print(f"Unique categories: {df['category_id'].nunique():,}")

    print(f"\nEarliest event time: {df['event_time'].min()}")
    print(f"Latest event time: {df['event_time'].max()}")

    print("\nDaily Behavior Statistics:")
    print(daily_stats.to_string())

    print(f"\nCleaned data saved to: {OUTPUT_PATH}")
    print(f"Daily statistics saved to: {DAILY_STATS_PATH}")

if __name__ == "__main__":
    main()