"""
Parse functions defintion file. Includes:
- FuncDef class with represents a function with name, description,
parameters and return types.
- parse_funcs() function with loads the function definitions
from a JSON file and stores them as a list of FuncDefs.
- FuncDefError which raises whenever there is an error
when parsing the functions defintion.
"""


import json
from pydantic import BaseModel, Field, model_validator, ValidationError


class FuncDefError(Exception):
    """Exception when there is an error parsing the functions defintion."""
    pass


class FuncDef(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]
    full_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validation(self) -> "FuncDef":
        """
        Verify that parameters and returns must have a type,
        and both keys and values cannot be empty.
        """
        valid_types = ["number", "string", "boolean", "integer"]
        for key, inner_dict in self.parameters.items():
            if len(key) == 0:
                raise ValueError("Parameter name cannot be empty.")
            if "type" not in inner_dict.keys():
                raise ValueError(f"Parameter '{key}' must have a type.")
            if inner_dict["type"].lower() not in valid_types:
                raise ValueError("Parameter types can only be "
                                 "'number', 'integer', 'string' or 'boolean'.")
            for inner_key, inner_val in inner_dict.items():
                if len(inner_key) == 0 or len(inner_val) == 0:
                    raise ValueError(f"Keys and values in parameter '{key}' "
                                     "cannot be empty")

        if len(self.returns) != 0 and "type" not in self.returns.keys():
            raise ValueError("Returns must have a type.")
        for key, val in self.returns.items():
            if len(key) == 0 or len(val) == 0:
                raise ValueError("Keys and values in 'returns' "
                                 "cannot be empty")

        return self


def parse_funcs(file: str) -> list[FuncDef]:
    """
    Reads from JSON file, validates the funtion defs
    and returns them in a list.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FuncDefError(f"FileNotFoundError: File {file} not found.")
    except PermissionError:
        raise FuncDefError(f"PermissionError: Cannot access file {file}.")
    except json.JSONDecodeError:
        raise FuncDefError(f"JSONDecodeError: File {file} "
                           "not in correct JSON format.")

    if len(data) == 0:
        raise FuncDefError(f"No function found in file {file}.")

    # Validate function defs in the file using pydantic obj
    funcs: list[FuncDef] = []
    for func in data:
        try:
            funcs.append(FuncDef(
                name=func["name"],
                description=func["description"],
                parameters=func["parameters"],
                returns=func["returns"],
                full_text=str(func)
            ))
        except KeyError:
            print(f"FunctionDefinitionError: Function defintion in '{func}' "
                  "must include name, description, parameters and returns")
            continue
        except ValidationError as e:
            print("FunctionDefinitionError: "
                  f"{e.errors()[0]["msg"]} in {func}")
            continue

    if len(funcs) > 0:
        return funcs
    else:
        raise FuncDefError("FunctionDefinitionError: "
                           "No valid function defined.")
