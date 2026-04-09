import pytest
from unittest.mock import MagicMock, patch
from src.parse_funcs import FuncDef
from src.parse_prompts import Prompt


def make_func_def(name: str,
                  description: str = "",
                  parameters: dict = {},
                  returns: dict = {},
                  full_text: str = "") -> FuncDef:
    """Helper to create a FuncDef with sensible defaults."""
    return FuncDef(
        name=name,
        description=description,
        parameters=parameters,
        returns=returns,
        full_text=full_text,
    )


def make_prompt(text: str) -> Prompt:
    """Helper to create a Prompt."""
    return Prompt(prompt=text)


def make_mock_llm(vocab_size: int = 100) -> MagicMock:
    """
    Create a mock Small_LLM_Model.
    encode() returns a mock tensor that supports .squeeze(0).tolist().
    decode() returns the string representation of the token ids.
    get_logits_from_input_ids() returns uniform logits by default.
    """
    llm = MagicMock()

    def mock_encode(text: str):
        """
        Encode text by mapping each character to its ASCII value.
        Wraps result in a mock tensor supporting .squeeze(0).tolist().
        """
        ids = [ord(c) for c in text]
        tensor = MagicMock()
        tensor.squeeze.return_value.tolist.return_value = ids
        return tensor

    def mock_decode(ids: list[int]) -> str:
        """Decode by converting ASCII values back to characters."""
        try:
            return "".join(chr(i) for i in ids
                           if 0 < i < 128)
        except (ValueError, TypeError):
            return ""

    def mock_logits(input_ids: list[int]) -> list[float]:
        """Return uniform logits — all tokens equally likely."""
        return [0.0] * vocab_size

    llm.encode.side_effect = mock_encode
    llm.decode.side_effect = mock_decode
    llm.get_logits_from_input_ids.side_effect = mock_logits

    return llm


@pytest.fixture
def mock_llm():
    return make_mock_llm()


@pytest.fixture
def sample_funcs():
    return [
        make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum",
            parameters={
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            returns={"type": "number"},
            full_text="def fn_add_numbers(a: float, b: float) -> float: ...",
        ),
        make_func_def(
            name="fn_greet",
            description="Generate a greeting message for a person by name.",
            parameters={
                "name": {"type": "string"},
            },
            returns={"type": "string"},
            full_text="def fn_greet(name: str) -> str: ...",
        ),
        make_func_def(
            name="fn_reverse_string",
            description="Reverse a string and return the reversed result.",
            parameters={
                "source_string": {
                    "type": "string"
                },
            },
            returns={"type": "string"},
            full_text="def fn_reverse_string(source_string: str) -> str: ...",
        ),
    ]
