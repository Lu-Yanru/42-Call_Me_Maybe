"""
write_output() function that takes a list and a file path
and writes the list to the file.
"""


import json
from pathlib import Path


def write_output(output: list, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "w") as f:
            json.dump(output, f)
    except (json.JSONDecodeError, TypeError, OSError):
        raise OSError(f"OSError: Failed to write to file {filepath}")
