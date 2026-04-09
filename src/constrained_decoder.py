"""
The ConstrainedDecoder class contains
basic functionalities of communicating with llm,
including tokenization, logit masking and match progress,
as well as the core constrained generation
using valid candidates.
"""


from functools import lru_cache


from llm_sdk import Small_LLM_Model
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class ConstrainedDecoder:
    """
    Shared tokenization utilities used by all components.
    Provides encoding, decoding, and candidate tokenization.
    """
    def __init__(self, llm: Small_LLM_Model, max_tokens: int = 150) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def generate_constrained(self, input_ids: list[int],
                             candidates: list[tuple[list[int], str]]) \
            -> str | None:
        """
        Core constrained generation loop.
        Generates freely until the first token of any candidate appears,
        then constrains generation to complete that candidate.
        Returns the matched candidate string or None on EOS/max_tokens.
        """
        # All generated tokens
        generated_ids: list[int] = []
        # Number of matched tokens for each candidate at matching index
        match_progress: list[int | None] = [None] * len(candidates)
        eos_ids = self.get_eos()

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Get the valid next token id
            valid_next = self.get_valid_next(candidates,
                                             match_progress)
            # Pick the best next token based on all generated token so far
            next_id, _ = self.get_next_token_id(input_ids + generated_ids,
                                                valid_next)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                return None

            generated_ids.append(next_id)
            matched = self.update_match_progress(next_id, candidates,
                                                 match_progress)
            if matched is not None:
                return matched

        return None

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
                          valid_ids: set[int] | None) -> tuple[int, float]:
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
            best_id = max(enumerate(masked_logits), key=lambda x: x[1])[0]
            return best_id, masked_logits[best_id]

        # Get the input id with the max logit
        best_id = max(enumerate(logits), key=lambda x: x[1])[0]
        return best_id, logits[best_id]

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
