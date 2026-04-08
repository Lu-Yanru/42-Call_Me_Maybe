from functools import lru_cache
import re
from typing import Any


from llm_sdk import Small_LLM_Model
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class PromptProcessor:
    def __init__(self, funcs: list[FuncDef], prompts: list[Prompt],
                 model_name: str, max_tokens: int = 150) -> None:
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
        Also includes \n.
        """
        eos_candidates = [
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "<eos>",
            "\n",
            " \n"
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
        prefix_len = len(self.encode_cache(prefix))

        res: list[tuple[list[int], str]] = []
        # Prevent duplicate candidates
        seen: set[tuple[int, ...]] = set()
        # Encode each function name into a list of token ids
        for can in candidates:
            token_ids_start = self.encode_cache(can)
            token_ids_mid = \
                self.encode_cache(prefix + can)[prefix_len:]
            # Also cases with explicit leading space
            token_ids_space = self.encode_cache(" " + can)

            for ids in [token_ids_mid, token_ids_start, token_ids_space]:
                key = tuple(ids)
                if ids and key not in seen:
                    seen.add(key)
                    res.append((ids, can))

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
            return max(enumerate(masked_logits), key=lambda x: x[1])[0]

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

        used_can: list[str] = []
        res: dict[str, Any] = {}
        if len(func_def.parameters) == 0:
            return res
        for var_name, param in func_def.parameters.items():
            if param["type"].lower() == "number":
                num_str = \
                    self.generate_num_param(prompt, func_def,
                                            var_name, param["type"],
                                            eos_ids, used_can)
                if num_str is not None:
                    try:
                        res[var_name] = float(num_str)
                        used_can.append(num_str)
                    except ValueError:
                        res[var_name] = None

            elif param["type"].lower() == "boolean":
                res[var_name] = \
                    self.generate_bool_param(prompt, func_def,
                                             var_name, param["type"],
                                             eos_ids)
            else:
                res[var_name] = \
                    self.generate_str_param(prompt, func_def,
                                            var_name, param["type"],
                                            eos_ids, used_can)
                if res[var_name] is not None:
                    used_can.append(res[var_name])

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
            message += " Answer in arabic number format. "
        elif type.lower() == "boolean":
            message += " Answer with 'true' or 'false'. "
        elif var_name.lower() == "regex":
            message += " Answer only with a valid regular expression. "

        message += f"Value of the parameter {var_name}: "
        return message

    def get_available_candidates(self, valid_candidates: list[str],
                                 used_candidates: list[str]) -> list[str]:
        """
        Remove already used candidates from the valid candidates
        to avoid repetition.
        """
        remaining_used = used_candidates.copy()
        available_candidates = []
        for n in valid_candidates:
            if n in remaining_used:
                remaining_used.remove(n)
            else:
                available_candidates.append(n)
        return available_candidates

    def count_parameters(self, func_def: FuncDef, type: str) -> int:
        """
        Count the total number of parameters of a certain type.
        """
        count = 0
        for param in func_def.parameters.values():
            if param["type"].lower() == type:
                count += 1
        return count

    def generate_num_param(self, prompt: Prompt, func_def: FuncDef,
                           var_name: str, type: str,
                           eos_ids: set[int],
                           used_candidates: list[str]) -> str | None:
        """
        Extract a numeric value of a parameter from the llm genertion
        using constrained decoding.
        Allow the llm to only generate numbers that appears in the prompt.
        Stops when a match is found or reaching max_tokens or EOS.

        Switch to free generating instead if
        there are fewer arabic numbers in the prompt than parameters.
        """
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        # Only allow numbers that appear in the prompt
        prompt_nums = self.get_valid_num(prompt.prompt)
        if len(prompt_nums) == 0:
            return self.generate_num_param_free(input_ids, eos_ids,
                                                used_candidates)
        # Switch to free generating if there are fewer arabic numbers
        # in the prompt than parameters
        total_num_parameters = self.count_parameters(func_def, "number")
        if len(prompt_nums) < total_num_parameters:
            return self.generate_num_param_free(input_ids, eos_ids,
                                                used_candidates)

        available_can = self.get_available_candidates(prompt_nums,
                                                      used_candidates)
        if len(available_can) == 0:
            return self.generate_num_param_free(input_ids, eos_ids,
                                                used_candidates)
        candidates = self.tokenize_str(available_can)

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
                return matched_number

        return None

    def generate_num_param_free(self, input_ids: list[int],
                                eos_ids: set[int],
                                used_can: list[str]) -> str | None:
        """
        If there are no arabic numbers in the prompt,
        extract the first arabic number
        generated by the llm using regex.
        Wait until the matched number stops growing before returning it.
        """
        # All generated tokens
        generated_ids: list[int] = []

        # Track the last matched number its length
        last_match: str = ""
        last_match_len = -1

        # Save the target number and stop re-evaluating which number to skip
        locked_target: str = ""

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             None)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                if len(last_match) > 0:
                    return last_match
                return None

            generated_ids.append(next_id)

            res = self.llm.decode(generated_ids)
            num = self.get_valid_num(res)
            if len(num) == 0:
                continue

            # Target number not yet identified
            if len(locked_target) == 0:
                # Skip the numbers in used_candidates
                remaining_can = used_can.copy()
                target_match = ""
                for n in num:
                    if n in remaining_can:
                        remaining_can.remove(n)
                    else:
                        target_match = n
                        break
                # If all found numbers so far have been used and thus skipped
                if len(target_match) == 0:
                    continue

                # Lock the target_match
                locked_target = target_match
                last_match = locked_target
                last_match_len = len(locked_target)

            # There is already a locked target match
            else:
                # Find match that starts with locked_target
                # and check if it still grows
                growing_match = next(
                    (n for n in num if n.startswith(locked_target) or
                     locked_target.startswith(n)),
                    None,
                )
                # Somehow no match is found
                if growing_match is None:
                    return last_match
                # Check if the number we match is still growing
                # and only return if it stops growing
                if len(growing_match) > last_match_len:
                    last_match = growing_match
                    last_match_len = len(target_match)
                    locked_target = growing_match
                # If the match is shorter than locked_target, keep generating
                elif locked_target.startswith(growing_match) \
                        and len(growing_match) < last_match_len:
                    continue
                else:
                    return last_match

        if len(last_match) > 0:
            return last_match
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

        # Only allow boolean values
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
                           var_name: str, type: str,
                           eos_ids: set[int],
                           used_candidates: list[str]) -> str | None:
        """
        Generate string parameters based on the function and the parameter.
        """
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        if var_name.lower() == "name":
            return self.generate_str_param_free(input_ids, eos_ids)

        # Only allow strings that appear in the prompt
        prompt_str = self.extract_string(prompt.prompt)

        if self.count_parameters(func_def, "string") > len(prompt_str):
            if "regex" in var_name.lower():
                return self.generate_regex_param(prompt, input_ids, eos_ids)
            if any(w in var_name.lower()
                   for w in ["replacement", "substitute"]):
                prompt_str = \
                    self.extract_replacement_candidates(prompt.prompt)
                # If there is only one candidate, just return it
                if len(prompt_str) == 1:
                    return prompt_str[0]

        available_can = self.get_available_candidates(prompt_str,
                                                      used_candidates)
        if len(available_can) == 0:
            return self.generate_str_param_free(input_ids, eos_ids)

        candidates = self.tokenize_str(available_can)

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

            # Update match progress and check if a string is fully matched
            res = self.update_match_progress(next_id, candidates,
                                             match_progress)

            if res is not None:
                return res.strip(" \'\"\n")

        return None

    def generate_str_param_free(self, input_ids: list[int],
                                eos_ids: set[int]) -> str:
        """
        Free generating with max logits.
        Fall back if there are no hints for valid tokens in the prompt
        for the parameters.
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

        res = self.llm.decode(generated_ids).strip()
        if "\n" in res:
            return res.split("\n")[0].strip("' \"")

        return res

    def generate_regex_param(self, prompt: Prompt, input_ids: list[int],
                             eos_ids: set[int]) -> str | None:
        """
        Generate a regex parameter by first selecting candidates
        from pre-made patterns in extract_regex_candidates()
        based on keywords found in the prompt.
        If no candidates or more than one candidates are found,
        use constrained decoding
        so the model can only use valid regex symbols.
        """
        # Get pre-made regex candidates based on prompt keywords
        regex_candidates = self.extract_regex_candidates(prompt.prompt.lower())

        # If there is only one candidate, just return it.
        if len(regex_candidates) == 1:
            return regex_candidates[0]

        # If no candidate matches, generate freely constrained by only
        # valid regex symbols
        if len(regex_candidates) == 0:
            return self.generate_regex_constrained(input_ids, eos_ids)

        # If multiple candidates match,
        # use constrained decoding with these as valid candidates
        candidates = self.tokenize_str(regex_candidates)
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

            # Update match progress and check if a string is fully matched
            res = self.update_match_progress(next_id, candidates,
                                             match_progress)

            if res is not None:
                return res.strip(" \'\"\n")

        return None

    def generate_regex_constrained(self, input_ids: list[int],
                                   eos_ids: set[int]) -> str | None:
        """
        Generate regex pattern by constraining to valid regex symbols.
        """
        generic_fallback = [str(n) for n in range(10)] \
            + [chr(c) for c in range(ord("a"), ord("z") + 1)] \
            + [chr(c) for c in range(ord("A"), ord("Z") + 1)] \
            + ["[", "]", "\\", "(", ")", "*", "?", "."] \
            + ["|", "^", "$", "+", "-", ",", "{", "}"] \
            + [">", "<", "'", '"', "!", ":", "#", "="]
        candidates = self.tokenize_str(generic_fallback)
        generated_ids: list[int] = []
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

            # Update match progress and check if a string is fully matched
            res = self.llm.decode(generated_ids).strip()
            if "\n" in res:
                return res.split("\n")[0].strip("' \"")

        return res

    def extract_string(self, string: str) -> list[str]:
        """
        Find all strings marked by '' or "" inside of a string.
        """
        return [content for _, content in
                re.findall(r"([\"'])(.*?)(?<!\\)\1", string)]

    def extract_regex_candidates(self, prompt: str) -> list[str]:
        """
        Premake some common regexes as candidates.
        """
        candidates = []
        # Vowels and consonants
        if "vowel" in prompt:
            if any(w in prompt for w in ["lowercase", "lower"]):
                # lowercase vowels only
                candidates.append(r"[aeiou]")
            elif any(w in prompt for w in ["uppercase", "upper", "capital"]):
                # uppercase vowels only
                candidates.append(r"[AEIOU]")
            else:
                # all vowels
                candidates.append(r"[aeiouAEIOU]")

        if "consonant" in prompt:
            if any(w in prompt for w in ["lowercase", "lower"]):
                candidates.append(r"[bcdfghjklmnpqrstvwxyz]")
            elif any(w in prompt for w in ["uppercase", "upper", "capital"]):
                candidates.append(r"[BCDFGHJKLMNPQRSTVWXYZ]")
            else:
                candidates.append(r"[bcdfghjklmnpqrstvwxyz"
                                  r"BCDFGHJKLMNPQRSTVWXYZ]")

        # Digits
        if any(w in prompt for w in ["non-digit", "non-number", "not digit",
                                     "not a digit", "not number",
                                     "not a number"]):
            candidates.append(r"\D+")
        elif any(w in prompt for w in ["alphanumeric", "alphabet or number",
                                       "a letter or a number",
                                       "letters and numbers"]):
            candidates.append(r"[a-zA-Z0-9]")
        elif any(w in prompt for w in ["digit", "number", "integer"]):
            candidates.append(r"\d+")

        # Letter patterns
        if any(w in prompt for w in ["uppercase letter", "upper letter",
                                     "capital letter", "uppercase alphabet",
                                     "uppercase character"]):
            candidates.append(r"[A-Z]")

        if any(w in prompt for w in ["lowercase letter", "lower letter",
                                     "lowercase alphabet",
                                     "lowercase character"]):
            candidates.append(r"[a-z]")

        # Only add general letter pattern if no specific case pattern was added
        # and the prompt mentions letters without specifying case
        if (not any(r in candidates for r in [r"[A-Z]", r"[a-z]"])
                and any(w in prompt for w in ["letter", "alphabet",
                                              "alphabetical"])):
            candidates.append(r"[a-zA-Z]")

        # Other patterns
        if any(w in prompt for w in ["space", "whitespace", "tab"]):
            candidates.append(r"\s+")

        if "punctuation" in prompt:
            candidates.append(r"[,.!?;:]")

        if any(w in prompt for w in ["special character", "special symbol"]):
            candidates.append(r"[^a-zA-Z0-9\s]")

        if "email" in prompt:
            candidates.append(r"\w+@\w+\.\w+")

        if any(w in prompt for w in ["url", "link"]):
            candidates.append(r"https?:\/\/\S+")
        return candidates

    def extract_replacement_candidates(self, prompt: str) -> list[str]:
        """
        Extract candidates for the replacement.
        Common formulations:
        Substitute/replace X with/by Y in/for 'Z'.
        Substitute/replace X in 'Z' with Y.
        Change/Turn X in 'Z' into Y.
        Change/Turn X into Y in 'Z'.
        In/For 'Z', replace X by/with Y (everywhere/globally).
        Convert X into/to Y
        """
        candidates: list[str] = []
        regex = (r"(?:with|into|by|to)\s+(.*?)"
                 r"(?:\s+(?:in|for|everywhere|globally)\b|[.!?]|$)")
        match = re.findall(regex, prompt, re.IGNORECASE)
        semantic_map = {
                        "asterisks": "*",
                        "asterisk": "*",
                        "stars": "*",
                        "star": "*",
                        "underscores": "_",
                        "underscore": "_",
                        "nothing": "",
                        "empty": "",
                        "blank": "",
                        "hyphens": "-",
                        "hyphen": "-",
                        "dash": "-",
                        "dashes": "-",
                        "minus": "-",
                        "space": " ",
                        "spaces": " ",
                        "whitespace": " ",
                        "whitespaces": " ",
                        "tab": "    ",
                        "dot": ".",
                        "dots": ".",
                    }
        for m in match:
            if m in semantic_map.keys():
                candidates.append(semantic_map[m])
            else:
                candidates.append(m.strip("\'\""))
        return candidates
