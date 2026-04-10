"""
The Visualizer class ochestrates
function name selection and parameter generation
while printing out messages on the terminal.
"""


import sys
from typing import Any


from llm_sdk import Small_LLM_Model
from src.generator_funcname import FuncNameGenerator
from src.generator_parameters import ParameterGenerator
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt
from src.process_prompt import ModelError


# Symbols and ANSI escape sequences for colors.
# ANSI format: \033[ = escape character, followed by a code, ending with 'm'
class Style:
    # Text colors
    GREEN = "\033[32m"   # success
    RED = "\033[31m"   # failure
    YELLOW = "\033[33m"   # warning / skipped
    CYAN = "\033[36m"   # info / prompt text
    WHITE = "\033[37m"   # neutral text
    BOLD = "\033[1m"    # bold text

    # Resets all styles back to terminal default
    RESET = "\033[0m"

    # Visual symbols used as status indicators
    CHECK = "✓"   # success marker
    CROSS = "✗"   # failure marker
    ARROW = "→"   # parameter assignment indicator
    DASH = "—"   # separator line character


class Visualizer:
    def __init__(self, funcs: list[FuncDef], prompts: list[Prompt],
                 model_name: str, max_tokens: int = 150,
                 use_color: bool = True) -> None:
        self.funcs = funcs
        self.prompts = prompts

        try:
            llm = Small_LLM_Model(model_name)
        except (ValueError, ImportError, OSError):
            raise ModelError(f"Failed to load model '{model_name}'.")
        except Exception:
            raise ModelError(f"Unexpected error laoding model '{model_name}'.")

        self.output: list[dict[str, str | dict[str, Any]]] = []
        self.funcname_generator = FuncNameGenerator(llm, funcs, max_tokens)
        self.param_generator = ParameterGenerator(llm, max_tokens)

        self.total_prompts = len(prompts)
        self.matched = 0
        self.skipped = 0
        # Whether to use ANSI color code.
        # Disable when output is piped to a file or non-color terminal
        self.use_color = use_color and sys.stdout.isatty()

    def process(self) -> None:
        """
        Create output using coalescence.
        """
        self.print_header()
        for index, prompt in enumerate(self.prompts):
            prompt_output: dict[str, str | dict[str, Any]] = {}

            prompt_output["prompt"] = prompt.prompt
            self.print_prompt(index + 1, prompt.prompt)

            func_name = self.funcname_generator.generate(prompt)
            if func_name is None:
                self.print_no_match()
                continue
            else:
                self.print_match(func_name)
                prompt_output["name"] = func_name

                for func in self.funcs:
                    if func.name == func_name:
                        func_def = func
                params = self.param_generator.generate(prompt, func_def)
                prompt_output["parameters"] = params
                for var_name, value in params.items():
                    self.print_parameter(var_name, value)

                self.output.append(prompt_output)
        self.print_summary()

    def style(self, text: str, *codes: str) -> str:
        """
        Wrap text in ANSI escape codes if color is enabled.
        Multiple codes can be added together.
        Always rests to style don't affect the next line.
        """
        if not self.use_color:
            return text
        return "".join(codes) + text + Style.RESET

    def separator(self, char: str = Style.DASH,
                  width: int = 60) -> str:
        """
        Create a separator line.
        """
        return self.style(char * width, Style.WHITE)

    def print_header(self) -> None:
        """
        Print a header.
        """
        print(self.separator())
        print(self.style(
            "📞 Call Me Maybe\n"
            f"Processing {self.total_prompts} prompts...",
            Style.BOLD, Style.CYAN
        ))
        print(self.separator())

    def print_prompt(self, index: int, prompt: str, max_len: int = 60) -> None:
        """
        Print the current prompt with index and truncates long prompt.
        """
        if len(prompt) > max_len:
            display = prompt[:max_len] + "..."
        else:
            display = prompt

        print(f"\n[{index}/{self.total_prompts}] "
              + self.style(f"Prompt: '{display}'", Style.CYAN))

    def print_match(self, func_name: str) -> None:
        """
        Print the function name if a match if found.
        """
        self.matched += 1
        print(
            "\t"
            + self.style(f"{Style.CHECK} Function: {func_name}", Style.GREEN)
        )

    def print_no_match(self) -> None:
        """
        Print message if no match was found.
        """
        self.skipped += 1
        print(
            "\t"
            + self.style(
                f"{Style.CROSS} No matching function found. Skipping...",
                Style.RED)
        )

    def print_parameter(self, var_name: str, value: Any) -> None:
        """
        Print the generated parameter.
        """
        if value is None:
            res = self.style("null", Style.YELLOW)
        else:
            res = self.style(str(value), Style.WHITE)

        print(
            "\t  "
            + self.style(f"{Style.ARROW} {var_name}: ", Style.WHITE)
            + res
            )

    def print_summary(self) -> None:
        """
        Print summary info.
        """
        print("")
        print(self.separator())
        print(self.style("Summary", Style.BOLD, Style.CYAN))
        print(self.separator())

        # Matched count
        print(
            self.style(f"{Style.CHECK} Matched: ", Style.GREEN)
            + self.style(str(self.matched), Style.GREEN)
            + self.style(f" / {self.total_prompts}", Style.WHITE)
        )

        # Skipped count
        if self.skipped > 0:
            print(
                self.style(f"{Style.CROSS} Skipped: ", Style.RED)
                + self.style(str(self.skipped), Style.RED)
                + self.style(f" / {self.total_prompts}", Style.WHITE)
            )

        print(self.separator())
