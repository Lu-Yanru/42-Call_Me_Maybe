"""
The BoolParamGenerator class is a subclass of ConstrainedDecoder
and handles the number parameter generation with constrained decoding.
"""


from llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class BoolParamGenerator(ConstrainedDecoder):
    def __init__(self, llm: Small_LLM_Model, max_token: int = 150) -> None:
        super().__init__(llm, max_token)

    def generate(self, prompt: Prompt, func_def: FuncDef,
                 var_name: str, type: str) -> bool | None:
        """
        Extract a boolean value of a parameter from the llm genertion
        Allow the llm to only generate true or false.
        Stops when a match is found or reaching max_tokens or EOS.
        """
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        # Only allow boolean values
        bool_vals = ["true", "false", "True", "False", "TRUE", "FALSE"]
        candidates = self.tokenize_str(bool_vals)

        matched = self.generate_constrained(input_ids, candidates)
        if matched is None:
            return None
        return matched.lower() == "true"
