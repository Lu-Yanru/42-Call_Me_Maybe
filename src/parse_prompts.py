"""
Parse input prompt file. Includes:
- Prompt class with represents a prompt.
- parse_prompts() function with loads the prompts
from a JSON file and stores them as a list of Prompt.
- PromptError which raises whenever there is an error
when parsing prompts.
"""


import json
from pydantic import BaseModel, Field, ValidationError


class PromptError(Exception):
    """Exception when there is an error parsing the prompts."""
    pass


class Prompt(BaseModel):
    prompt: str = Field(min_length=1)


def parse_prompts(file: str) -> list[Prompt]:
    """
    Reads from JSON file, validates the prompts
    and returns them in a list.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise PromptError(f"FileNotFoundError: File {file} not found.")
    except PermissionError:
        raise PromptError(f"PermissionError: Cannot access file {file}.")
    except json.JSONDecodeError:
        raise PromptError(f"JSONDecodeError: File {file} "
                          "not in correct JSON format.")

    if len(data) == 0:
        raise PromptError(f"No prompt found in file {file}.")

    prompts: list[Prompt] = []
    # Validate function defs in the file using pydantic obj
    for prompt in data:
        try:
            prompts.append(Prompt(
                prompt=prompt["prompt"],
            ))
        except KeyError:
            print(f"PromptError: Prompt not defined in '{prompt}'.")
            continue
        except ValidationError as e:
            print(f"PromptError: {e.errors()[0]["msg"]} in '{prompt}'")
            continue

    if len(prompts) > 0:
        return prompts
    else:
        raise PromptError("PromptError: No valid prompt.")
