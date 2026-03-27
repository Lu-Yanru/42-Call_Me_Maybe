from typing import Any


from llm_sdk import Small_LLM_Model
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
            prompt_output["response"] = self.generate(prompt)
            # prompt_output["name"] = self.generate_func_name(prompt)
            # prompt_output["parameters"] = self.generate_parameters(prompt)

            self.output.append(prompt_output)

    def get_next_token_id(self, input_ids: list[int]) -> int:
        # Get a list of logits of the input ids
        logits = self.llm.get_logits_from_input_ids(input_ids)

        # Get the input id with the max logit
        return max(enumerate(logits), key=lambda x: x[1])[0]

    def generate(self, prompt: Prompt) -> str:
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.llm.encode(prompt.prompt).squeeze(0).tolist()

        generated_ids: list[int] = []

        for _ in range(10):
            # Feed the full ids into the llm
            next_id = self.get_next_token_id(input_ids + generated_ids)
            generated_ids.append(next_id)

        return self.llm.decode(generated_ids)

    # def generate_func_name(self, prompt: Prompt) -> str:
    #     pass
