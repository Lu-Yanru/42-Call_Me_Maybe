"""
The FuncNameGenerator class is a subclass of ConstrainedDecoder
and handles the function name generation with constrained decoding.
"""


from llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class FuncNameGenerator(ConstrainedDecoder):
    def __init__(self, llm: Small_LLM_Model, funcs: list[FuncDef],
                 max_token: int = 150) -> None:
        super().__init__(llm, max_token)
        self.funcs = funcs

    def generate(self, prompt: Prompt) -> str | None:
        """
        Use constrained decoding to make the llm pick one function name
        from the provided function definitions.
        Generate until a valid function name is found,
        or when a maximun number of tokens is reached.
        """
        message = self.create_prompt(prompt)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        func_names = self.get_func_names()
        candidates = self.tokenize_str(func_names)

        return self.generate_constrained(input_ids, candidates)

    def get_func_defs(self) -> list[dict[str, str]]:
        """
        Create a list of dict with only the function names and descriptions.
        """
        res = []
        for func in self.funcs:
            res.append(
                {"name": func.name,
                 "description": func.description}
            )
        return res

    def get_func_names(self) -> list[str]:
        """Create a list of function names"""
        func_names = []
        for func in self.funcs:
            func_names.append(func.name)
        return func_names

    def create_prompt(self, prompt: Prompt) -> str:
        """
        Create a prompt using the input prompt
        and the function definitions.
        """
        func_defs = self.get_func_defs()
        message = ("Here is a list of functions "
                   "each with a name and a description of what it does: "
                   f"{func_defs}"
                   "Select the most appropriate function among the list above "
                   f"to solve the task '{prompt.prompt}'. "
                   "Answer with only the name of the selected function.")
        return message
