"""
Parse functions defintion file. Includes:
- FuncDef class with represents a function with name, description, parameters and return types.
- parse_funcs() function with loads the function definitions
from a JSON file and stores them as a list of FuncDefs.
- FuncDefError which raises whenever there is an error when parsing the functions defintion.
"""


import json
from pydantic import BaseModel, model_validator


class FuncDefError(Exception):
    """Exception when there is an error parsing the functions defintion."""
    pass


class FuncDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]

    @model_validator(mode="after")
    def validation(self) -> "FuncDef":
        return self


def parse_funcs(file: str) -> list[FuncDef]:
    try:
        with open(file, "r") as f:
            data = json.load(f)
        if len(data) == 0:
            raise FuncDef(f"No function found in file {file}.")
    except FileNotFoundError:
        raise FuncDefError(f"File {file} not found.")
