"""
The ParameterGenerator class is a subclass of ConstrainedDecoder
and handles the parameter generation with constrained decoding.
"""


from typing import Any


from llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.generator_param_bool import BoolParamGenerator
from src.generator_param_num import NumParamGenerator
from src.generator_param_str import StrParamGenerator
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class ParameterGenerator(ConstrainedDecoder):
    def __init__(self, llm: Small_LLM_Model, max_tokens: int = 150) -> None:
        super().__init__(llm, max_tokens)
        self.num_param = NumParamGenerator(llm, max_tokens)
        self.bool_param = BoolParamGenerator(llm, max_tokens)
        self.str_param = StrParamGenerator(llm, max_tokens)

    def generate(self, prompt: Prompt,
                 func_def: FuncDef) -> dict[str, Any]:
        """
        Use constrained decoding to make the llm
        pick the parameters of the function
        based on the prompt and the function description.
        """
        used_can: list[str] = []
        res: dict[str, Any] = {}
        if len(func_def.parameters) == 0:
            return res
        for var_name, param in func_def.parameters.items():
            if param["type"].lower() == "number":
                num_str = \
                    self.num_param.generate(prompt, func_def,
                                            var_name, param["type"],
                                            used_can)
                if num_str is not None:
                    try:
                        res[var_name] = float(num_str)
                        used_can.append(num_str)
                    except ValueError:
                        res[var_name] = None

            elif param["type"].lower() == "integer":
                num_str = \
                    self.num_param.generate(prompt, func_def,
                                            var_name, param["type"],
                                            used_can)
                if num_str is not None:
                    try:
                        res[var_name] = int(num_str)
                        used_can.append(num_str)
                    except ValueError:
                        res[var_name] = None

            elif param["type"].lower() == "boolean":
                res[var_name] = \
                    self.bool_param.generate(prompt, func_def,
                                             var_name, param["type"])
            else:
                res[var_name] = \
                    self.str_param.generate(prompt, func_def,
                                            var_name, param["type"],
                                            used_can)
                if res[var_name] is not None:
                    used_can.append(res[var_name])

        return res
