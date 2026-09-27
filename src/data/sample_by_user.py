from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "UserBehavior.csv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "user_behavior_sample.parquet"
)

COLUMNS = [
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
    "timestamp"
]

DTYPES = {
    "user_id": "int64",
    "item_id": "int64",
    "category_id": "int64",
    "behavior_type": "string",
    "timestamp": "int64"
}

SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("item_id", pa.int64()),
        ("category_id", pa.int64()),
        ("behavior_type", pa.string()),
        ("timestamp", pa.int64()),
    ]
)

SAMPLE_MOD = 1000
SAMPLE_THRESHOLD = 20
CHUNK_SIZE = 1_000_000

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Can't find raw data:{INPUT_PATH}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Output file already exists:{OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        OUTPUT_PATH,
        schema=SCHEMA,
        compression="snappy",
    )

    total_rows = 0
    sampled_rows = 0
    chunk_number = 0

    try:
        reader = pd.read_csv(
            INPUT_PATH,
            names=COLUMNS,
            dtype=DTYPES,
            chunksize=CHUNK_SIZE
        )

        for chunk in reader:
            chunk_number += 1
            total_rows += len(chunk)

            mask = chunk["user_id"].mod(SAMPLE_MOD) < SAMPLE_THRESHOLD
            sampled = chunk.loc[mask, COLUMNS]

            if not sampled.empty:
                table = pa.Table.from_pandas(
                    sampled,
                    schema=SCHEMA,
                    preserve_index=False,
                )
                writer.write_table(table)
                sampled_rows += len(sampled)

                print(
                    f"Totally {chunk_number} chunks",
                    f"Read {total_rows:,} rows",
                    f"Preserved {sampled_rows} rows"
                )

    finally:
        writer.close()

    sample_rate = sampled_rows / total_rows if total_rows else 0

    print(f"Total raw interactions: {total_rows}")
    print(f"Sampled interactions: {sampled_rows}")
    print(f"Actual sample rate: {sample_rate: .2%}")

if __name__ == "__main__":
    main()