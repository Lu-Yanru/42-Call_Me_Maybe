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
            prompt_output["name"] = self.generate_func_name(prompt)
            # prompt_output["parameters"] = self.generate_parameters(prompt)

            self.output.append(prompt_output)

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

    def tokenize_func_names(self) -> list[tuple[list[int], str]]:
        """
        Create a list of (token_id, function_name) pairs.
        Each function name matches a start-of-sentence token version
        and a mid-sentence token version.
        """
        prefix = "A "
        prefix_len = len(self.llm.encode(prefix).squeeze(0).tolist())

        func_names = self.get_func_names()

        res: list[tuple[list[int], str]] = []
        # Encode each function name into a list of token ids
        for name in func_names:
            token_ids_start = self.llm.encode(name).squeeze(0).tolist()
            token_ids_mid = \
                self.llm.encode(prefix + name).squeeze(0).tolist()[prefix_len:]
            res.append((token_ids_mid, name))
            if token_ids_mid != token_ids_start:
                res.append((token_ids_start, name))

        return res

    def get_valid_func_name_id(self, candidates: list[tuple[list[int], str]],
                               match_progress: list[int | None]) \
            -> set[int] | None:
        """
        Returns the set of valid next token ids from all active function name
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
        Update the match_progress for each candidate function name.
        Returns the matched function name if any candidate is fully matched,
        None otherwise.
        """
        for i, (token_ids, name) in enumerate(candidates):
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

    def generate_func_name(self, prompt: Prompt, max_tokens: int = 500) -> str:
        """
        Use constrained decoding to make the llm pick one function name
        from the provided function definitions.
        Generate until a valid function name is found,
        or when a maximun number of tokens is reached.
        """
        message = self.create_prompt_func_name(prompt)

        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.llm.encode(message).squeeze(0).tolist()

        candidates = self.tokenize_func_names()

        # All generated tokens
        generated_ids: list[int] = []
        # Number of matched tokens for each candidate at matching index
        match_progress: list[int | None] = [None] * len(candidates)

        # Generate one token at a time, up to max_tokens
        for _ in range(max_tokens):
            # Get the valid next token id
            valid_next = self.get_valid_func_name_id(candidates,
                                                     match_progress)
            # Pick the best next token based on all generated token so far
            next_id = self.get_next_token_id(input_ids + generated_ids,
                                             valid_next)
            generated_ids.append(next_id)
            matched_name = self.update_match_progress(next_id, candidates,
                                                      match_progress)
            if matched_name is not None:
                return matched_name

        return "No match found"
