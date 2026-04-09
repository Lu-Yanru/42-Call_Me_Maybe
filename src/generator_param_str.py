"""
The StrParamGenerator class is a subclass of ConstrainedDecoder
and handles the number parameter generation with constrained decoding.
"""


import re


from llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


class StrParamGenerator(ConstrainedDecoder):
    def __init__(self, llm: Small_LLM_Model, max_token: int = 150) -> None:
        super().__init__(llm, max_token)

    def generate(self, prompt: Prompt, func_def: FuncDef,
                 var_name: str, type: str,
                 used_candidates: list[str]) -> str | None:
        """
        Generate string parameters based on the function and the parameter.
        """
        message = self.create_prompt_parameters(prompt, func_def,
                                                var_name, type)
        # Encode prompt into a 2D tensor and convert into a list
        input_ids = self.encode_cache(message)

        if var_name.lower() == "name":
            return self.generate_str_param_free(input_ids)

        # Only allow strings that appear in the prompt
        prompt_str = self.extract_string(prompt.prompt)

        if self.count_parameters(func_def, "string") > len(prompt_str):
            if "regex" in var_name.lower():
                return self.generate_regex_param(prompt, input_ids)
            if any(w in var_name.lower()
                   for w in ["replacement", "substitute"]):
                prompt_str = \
                    self.extract_replacement_candidates(prompt.prompt)
                # If there is only one candidate, just return it
                if len(prompt_str) == 1:
                    return prompt_str[0]
        # If there are enough quoted strings
        # parameter "source_string" prefers the longest string
        else:
            if "source" in var_name.lower():
                available_can = self.get_available_candidates(prompt_str,
                                                              used_candidates)
                if len(available_can) > 0:
                    longest = max(available_can, key=len)
                    return longest

        available_can = self.get_available_candidates(prompt_str,
                                                      used_candidates)
        if len(available_can) == 0:
            return self.generate_str_param_free(input_ids)

        candidates = self.tokenize_str(available_can)

        matched = self.generate_constrained(input_ids, candidates)
        if matched is None:
            return None
        return matched.strip(" \'\"")

    def generate_str_param_free(self, input_ids: list[int]) -> str:
        """
        Free generating with max logits.
        Fall back if there are no hints for valid tokens in the prompt
        for the parameters.
        """
        eos_ids = self.get_eos()
        # All generated tokens
        generated_ids: list[int] = []

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            # Pick the best next token based on all generated token so far
            next_id, _ = self.get_next_token_id(input_ids + generated_ids,
                                                None)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                break

            generated_ids.append(next_id)

        res = self.llm.decode(generated_ids).strip()
        if "\n" in res:
            return res.split("\n")[0].strip("' \"")

        return res

    def generate_regex_param(self, prompt: Prompt,
                             input_ids: list[int]) -> str | None:
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

        # If multiple candidates match,
        # use constrained decoding with these as valid candidates
        if len(regex_candidates) > 1:
            candidates = self.tokenize_str(regex_candidates)
            matched = self.generate_constrained(input_ids, candidates)
            if matched is not None:
                return matched.strip(" \'\"\n")

        # If no candidate matches, generate freely constrained by only
        # valid regex symbols
        return self.generate_regex_constrained(input_ids)

    def generate_regex_constrained(self, input_ids: list[int]) -> str | None:
        """
        Generate regex pattern by constraining to valid regex symbols.
        Constrain to only generate valid symbols,
        stop as soon as the next token is not valid anymore
        and returns the generated tokens so far.
        """
        generic_fallback = [str(n) for n in range(10)] \
            + [chr(c) for c in range(ord("a"), ord("z") + 1)] \
            + [chr(c) for c in range(ord("A"), ord("Z") + 1)] \
            + ["[", "]", "\\", "(", ")", "*", "?", "."] \
            + ["|", "^", "$", "+", "-", ",", "{", "}"] \
            + [">", "<", "'", '"', "!", ":", "#", "="]
        # Build the set of valid token ids from the generic_fallback chars
        valid_ids: set[int] = {
            ids[0]
            for ids, _ in self.tokenize_str(generic_fallback)
            if len(ids) == 1  # Only single token chars are valid
        }

        eos_ids = self.get_eos()
        generated_ids: list[int] = []

        # Generate one token at a time, up to max_tokens
        for _ in range(self.max_tokens):
            next_id, next_logit = self.get_next_token_id(
                input_ids + generated_ids, valid_ids)
            # Stop if llm generates EOS
            if next_id in eos_ids:
                break

            # Stop if the next_id is invalid (logit is -inf)
            if next_logit == float("-inf"):
                break

            generated_ids.append(next_id)

        return self.llm.decode(generated_ids).strip(" \'\"")

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
            # If match is one of the keywords above, use the symbol
            if m in semantic_map.keys():
                candidates.append(semantic_map[m])
            # If not, check if there is a quoted string inside
            # If yes, use the quoted string
            # Otherwise use the whole match
            else:
                strs = self.extract_string(m)
                if len(strs) == 0:
                    candidates.append(m.strip(" \'\""))
                else:
                    candidates.append(strs[0].strip("\'\""))
        return candidates
