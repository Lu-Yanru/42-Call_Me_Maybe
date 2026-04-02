"""
write_output() function that takes a list and a file path
and writes the list to the file.
"""


import json
from pathlib import Path
from typing import Any


def write_output(output: list[dict[str, str | dict[str, Any]]],
                 filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    except (json.JSONDecodeError, TypeError, OSError):
        raise OSError(f"OSError: Failed to write to file {filepath}")
