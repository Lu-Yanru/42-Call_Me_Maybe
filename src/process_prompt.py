from functools import lru_cache
import re
from typing import Any


from llm_sdk import Small_LLM_Model
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class PromptProcessor:
    def __init__(self, funcs: list[FuncDef], prompts: list[Prompt],
                 model_name: str, max_tokens: int = 200) -> None:
        self.funcs = funcs
        self.prompts = prompts
        self.llm = Small_LLM_Model(model_name)
        self.output: list[dict[str, str | dict[str, Any]]] = []
        self.max_tokens = max_tokens

    def process(self) -> None:
        """
        Create output using coalescence.
        """
        for prompt in self.prompts:
            prompt_output: dict[str, str | dict[str, Any]] = {}

            prompt_output["prompt"] = prompt.prompt
            func_name = self.generate_func_name(prompt)
            if func_name is None:
                print("Cannot find a suitable function for "
                      f"'{prompt.prompt}'. "
                      "Skipping...")
                continue
            else:
                prompt_output["name"] = func_name
                prompt_output["parameters"] = \
                    self.generate_parameters(prompt, func_name)

                self.output.append(prompt_output)

    @lru_cache
    def get_eos(self) -> set[int]:
        """
        Get the token id of eos by checking various way of encoding eos.
        """
        eos_candidates = [
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "<eos>"
        ]
        eos_ids: set[int] = set()
        for str in eos_candidates:
            ids = self.llm.encode(str).squeeze(0).tolist()
            if len(ids) == 1 and self.llm.decode(ids) == "":
                eos_ids.add(ids[0])
        return eos_ids

    @lru_cache
    def encode_cache(self, text: str) -> list[int]:
        return self.llm.encode(text).squeeze(0).tolist()

    def tokenize_str(self, candidates: list[str]) \
            -> list[tuple[list[int], str]]:
        """
        Create a list of (token_id, candidate) pairs.
        Each candidate (func_name or number) matches
        a start-of-sentence token version
        and a mid-sentence token version.
        """
        prefix = "A "
        prefix_len = len(self.llm.encode(prefix).squeeze(0).tolist())

        res: list[tuple[list[int], str]] = []
        # Encode each function name into a list of token ids
        for can in candidates:
            token_ids_start = self.encode_cache(can)
            token_ids_mid = \
                self.encode_cache(prefix + can)[prefix_len:]
            if token_ids_mid:
                res.append((token_ids_mid, can))
            if token_ids_start and token_ids_mid != token_ids_start:
                res.append((token_ids_start, can))

        return res

    def get_valid_next(self, candidates: list[tuple[list[int], str]],
                       match_progress: list[int | None]) \
            -> set[int] | None:
        """
        Returns the set of valid next token ids from all
        candidates. None if no candidate is active.
        """
        valid_next: set[int] = set()
        for (token_ids, _), progress in zip(candidates, match_progress):
            if progress is not None and progress < len(token_ids):
                valid_next.add(token_ids[progress])
        if len(valid_next) > 0:
            return valid_next
        else:
            return None

    def get_next_token_id(self, input_ids: list[int],
                          valid_ids: set[int] | None) -> int:
        """
        Get the token id with the max logit,
        restricted to only valid tokens.
        All other logits are masked to -inf.
        """
        # Get a list of logits of the input ids
        logits = self.llm.get_logits_from_input_ids(input_ids)

        if valid_ids is not None:
            # Mask logits of invalid tokens to -inf
            masked_logits: list[float] = []
            for id, logit in enumerate(logits):
                if id in valid_ids:
                    masked_logits.append(logit)
                else:
                    masked_logits.append(float("-inf"))

        # Get the input id with the max logit
        return max(enumerate(logits), key=lambda x: x[1])[0]

    def update_match_progress(self, next_id: int,
                              candidates: list[tuple[list[int], str]],
                              match_progress: list[int | None]) -> str | None:
        """
        Update the match_progress for each candidate function name/parameter.
        Returns the matched function name if any candidate is fully matched,
        None otherwise.
        """
        for i, (token_ids, name) in enumerate(candidates):
            # Skip candidates with empty token lists
            if not token_ids:
                continue

            progress = match_progress[i]

            # If nothing is matched yet,
            # check if next_id matches the first token
            if progress is None:
                if next_id == token_ids[0]:
                    match_progress[i] = 1
            # If something is matched already,
            # check if next_id matches the next in token_ids
            else:
                if progress < len(token_ids):
                    expected = token_ids[progress]
                else:
                    expected = None
                if next_id == expected:
                    match_progress[i] = progress + 1
                else:
                    match_progress[i] = None

            # Check if this candidate has been fully matched
            if match_progress[i] == len(token_ids):
                return name

        return None

    def generate_func_name(self, prompt: Prompt) -> str | None:
        """
        Use constrained decoding to make the llm pick one function name
        from the provided function definitions.
        Generate until a valid function name is found,
        or when a maximun number of tokens is reached.
        """
        message = self.create_prompt_func_name(prompt)
        eos_ids = self.get_eos()

        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        func_names = self.get_func_names()
        candidates = self.tokenize_str(func_names)

        # All generated tokens
        generated_ids: list[int] = []
        # Number of matched tokens for each candidate at matching index
        match_progress: list[int | None] = [None] * len(candidates)

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Get the valid next token id
            valid_next = self.get_valid_next(candidates,
                                             match_progress)
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             valid_next)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                return None

            generated_ids.append(next_id)
            matched_name = self.update_match_progress(next_id, candidates,
                                                      match_progress)
            if matched_name is not None:
                return matched_name

        return None

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

    def create_prompt_func_name(self, prompt: Prompt) -> str:
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

    def generate_parameters(self, prompt: Prompt,
                            func_name: str) -> dict[str, Any]:
        """
        Use constrained decoding to make the llm
        pick the parameters of the function
        based on the prompt and the function description.
        """
        for func in self.funcs:
            if func.name == func_name:
                func_def = func
        eos_ids = self.get_eos()

        res: dict[str, Any] = {}
        if len(func_def.parameters) == 0:
            return res
        for var_name, param in func_def.parameters.items():
            if param["type"].lower() == "number":
                res[var_name] = \
                    self.generate_num_param(prompt, func_def,
                                            var_name, param["type"],
                                            eos_ids)
            elif param["type"].lower() == "boolean":
                res[var_name] = \
                    self.generate_bool_param(prompt, func_def,
                                             var_name, param["type"],
                                             eos_ids)
            else:
                res[var_name] = \
                    self.generate_str_param(prompt, func_def,
                                            var_name, param["type"],
                                            eos_ids)

        return res

    def create_prompt_parameters(self, prompt: Prompt,
                                 func_def: FuncDef,
                                 var_name: str,
                                 type: str) -> str:
        """
        Creates a prompt message that asks the llm
        to generate the value of a parameter for the input prompt
        based on the given function.
        """
        message = (f"Use the following function: {func_def.full_text} "
                   f"to solve the task '{prompt.prompt}'. "
                   "Do not give the answer to the task directly. "
                   f"Provide only the value of the parameter {var_name} "
                   f"of type {type}, "
                   "and nothing else.")
        if type.lower() == "number":
            message += (" Answer with arabic numerals. "
                        f"Value of the parameter {var_name}: ")
        elif type.lower() == "boolean":
            message += (" Answer with 'true' or 'false'. "
                        f"Value of the parameter {var_name}: ")
        else:
            message += f"Value of the parameter {var_name}: "
        return message

    def generate_num_param(self, prompt: Prompt, func_def: FuncDef,
                           var_name: str, type: str,
                           eos_ids: set[int]) -> int | float | None:
        """
        Extract a numeric value of a parameter from the llm genertion
        using constrained decoding.
        Allow the llm to only generate numbers that appears in the prompt.
        Stops when a match is found or reaching max_tokens or EOS.
        """
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        # Only allow numbers that appear in the prompt
        prompt_nums = self.get_valid_num(prompt.prompt)
        if len(prompt_nums) == 0:
            return self.generate_num_param_free(input_ids, eos_ids)
        candidates = self.tokenize_str(prompt_nums)

        # All generated tokens
        generated_ids: list[int] = []
        # Number of matched tokens for each candidate at matching index
        match_progress: list[int | None] = [None] * len(candidates)

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Get the valid next token id
            valid_next = self.get_valid_next(candidates,
                                             match_progress)
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             valid_next)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                break

            generated_ids.append(next_id)

            # Update match progress and check if a number is fully matched
            matched_number = self.update_match_progress(next_id, candidates,
                                                        match_progress)

            if matched_number is not None:
                return float(matched_number)

        return None

    def generate_num_param_free(self, input_ids: list[int],
                                eos_ids: set[int]) -> int | float | None:
        """
        If there are no arabic numbers in the prompt,
        extract the first arabic number
        generated by the llm using regex.
        """
        # All generated tokens
        generated_ids: list[int] = []

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             None)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                break

            generated_ids.append(next_id)

            res = self.llm.decode(generated_ids)
            num = self.get_valid_num(res)
            if len(num) > 0:
                return float(num[0])

        return None

    def get_valid_num(self, string: str) -> list[str]:
        """
        Extract all numbers strings that appear in a string
        using regex.
        """
        return re.findall(r"-?\d+(?:\.\d+)?", string)

    def generate_bool_param(self, prompt: Prompt, func_def: FuncDef,
                            var_name: str, type: str,
                            eos_ids: set[int]) -> bool | None:
        """
        Extract a boolean value of a parameter from the llm genertion
        Allow the llm to only generate true or false.
        Stops when a match is found or reaching max_tokens or EOS.
        """
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        # Only allow numbers that appear in the prompt
        bool_vals = ["true", "false", "True", "False", "TRUE", "FALSE"]
        candidates = self.tokenize_str(bool_vals)

        # All generated tokens
        generated_ids: list[int] = []
        # Number of matched tokens for each candidate at matching index
        match_progress: list[int | None] = [None] * len(candidates)

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Get the valid next token id
            valid_next = self.get_valid_next(candidates,
                                             match_progress)
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             valid_next)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                break

            generated_ids.append(next_id)

            # Update match progress and check if a number is fully matched
            res = self.update_match_progress(next_id, candidates,
                                             match_progress)

            if res is not None:
                return res.lower() == "true"

        return None

    def generate_str_param(self, prompt: Prompt, func_def: FuncDef,
                           var_name: str, type: str, eos_ids: set[int]) -> str:
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)

        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        # All generated tokens
        generated_ids: list[int] = []

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             None)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                break

            generated_ids.append(next_id)

        return self.llm.decode(generated_ids).strip()
