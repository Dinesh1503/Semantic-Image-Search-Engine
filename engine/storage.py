import json
from datetime import datetime

DEFAULT_DATA_FILE = "data.json"


def save_records(records: list, output_path: str = DEFAULT_DATA_FILE):
    """Persist (name, path, caption, embedding, captured_at) rows as JSON."""
    serialisable = [
        [
            name,
            path,
            caption,
            list(embedding),
            captured_at.isoformat() if isinstance(captured_at, datetime) else captured_at,
        ]
        for name, path, caption, embedding, captured_at in records
    ]
    with open(output_path, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_records(input_path: str = DEFAULT_DATA_FILE) -> list:
    """Load rows saved by `save_records` in the tuple shape `VectorDatabase.add_data` expects."""
    with open(input_path, "r") as f:
        raw = json.load(f)

    return [
        (row[0], row[1], row[2], row[3], json.dumps({"datetime": row[4]}))
        for row in raw
    ]
