"""
The PromptProcessor class is the top-level ochestrator
that assembles function name selection and parameter generation
for a list of prompts.
"""


from typing import Any


from llm_sdk import Small_LLM_Model
from src.generator_funcname import FuncNameGenerator
from src.generator_parameters import ParameterGenerator
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class PromptProcessor:
    def __init__(self, funcs: list[FuncDef], prompts: list[Prompt],
                 model_name: str, max_tokens: int = 150) -> None:
        self.funcs = funcs
        self.prompts = prompts
        llm = Small_LLM_Model(model_name)
        self.output: list[dict[str, str | dict[str, Any]]] = []
        self.funcname_generator = FuncNameGenerator(llm, funcs, max_tokens)
        self.param_generator = ParameterGenerator(llm, max_tokens)

    def process(self) -> None:
        """
        Create output using coalescence.
        """
        for prompt in self.prompts:
            prompt_output: dict[str, str | dict[str, Any]] = {}

            prompt_output["prompt"] = prompt.prompt
            func_name = self.funcname_generator.generate(prompt)
            if func_name is None:
                print("Cannot find a suitable function for "
                      f"'{prompt.prompt}'. "
                      "Skipping...")
                continue
            else:
                prompt_output["name"] = func_name

                for func in self.funcs:
                    if func.name == func_name:
                        func_def = func
                prompt_output["parameters"] = \
                    self.param_generator.generate(prompt, func_def)

                self.output.append(prompt_output)
