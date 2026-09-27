from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UserBehavior.csv"

COLUMNS = [
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
    "timestamp",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}")

    # 第一次只检查前10万行，避免直接加载全部数据
    df = pd.read_csv(
        DATA_PATH,
        names=COLUMNS,
        nrows=100_000,
        dtype={
            "user_id": "int64",
            "item_id": "int64",
            "category_id": "int64",
            "behavior_type": "category",
            "timestamp": "int64",
        },
    )

    df["event_time"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Shanghai")
    )

    print("样本行数：", len(df))
    print("\n前5行：")
    print(df.head())

    print("\n字段类型：")
    print(df.dtypes)

    print("\n空值数量：")
    print(df.isna().sum())

    print("\n重复记录数：")
    print(df.duplicated().sum())

    print("\n行为类型分布：")
    print(df["behavior_type"].value_counts(dropna=False))

    print("\n用户数：", df["user_id"].nunique())
    print("商品数：", df["item_id"].nunique())
    print("类别数：", df["category_id"].nunique())

    print("\n最早时间：", df["event_time"].min())
    print("最晚时间：", df["event_time"].max())

    valid_behaviors = {"pv", "fav", "cart", "buy"}
    actual_behaviors = set(df["behavior_type"].dropna().astype(str))
    invalid_behaviors = actual_behaviors - valid_behaviors

    print("\n非法行为类型：", invalid_behaviors or "无")


if __name__ == "__main__":
    main()