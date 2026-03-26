from typing import Any


from llm_sdk.llm_sdk import Small_LLM_Model
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class PromptProcessor:
    def __init__(self, funcs: list[FuncDef], prompts: list[Prompt],
                 model_name: str) -> None:
        self.funcs = funcs
        self.prompts = prompts
        self.llm = Small_LLM_Model(model_name)
        self.output: list[dict[str, str | dict[str, Any]]] = []

    def process(self) -> None:
        """
        Create output using coalescence.
        """
        for prompt in self.prompts:
            prompt_output: dict[str, str | dict[str, Any]] = {}

            prompt_output["output"] = prompt.prompt
            prompt_output["name"] = self.generate_func_name(prompt)
            prompt_output["parameters"] = self.generate_parameters(prompt)

            self.output.append(prompt_output)

    def generate_func_name(self, prompt: Prompt) -> str:
        pass
