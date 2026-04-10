import pytest
from unittest.mock import MagicMock
from src.process_prompt import ParameterGenerator
from src.generator_funcname import FuncNameGenerator
from src.generator_param_bool import BoolParamGenerator
from src.generator_param_num import NumParamGenerator
from src.generator_param_str import StrParamGenerator
from tests.conftest import make_func_def, make_prompt, make_mock_llm


# ------------------------------------------------------------------ #
# NumParamGenerator tests                                          #
# ------------------------------------------------------------------ #

class TestNumParamGenerator:

    def test_returns_float_when_number_in_prompt(self):
        """
        When the prompt contains an arabic number and the model generates
        its tokens, generate() should return the number as a float.
        """
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=100)
        func_def = make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
                },
            returns={
                "type": "number"
            },
            full_text="a"
        )
        prompt = make_prompt("Add 3 and 7")

        # Make model generate tokens for "3"
        target_ids = gen.encode_cache("3")
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(target_ids):
                logits[target_ids[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0  # EOS
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)

        result = gen.generate(prompt, func_def, "a", "number", [])
        assert float(result) == 3.0

    def test_returns_none_when_no_number_generated(self):
        """When EOS is generated immediately, return None."""
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        func_def = make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
                },
            returns={
                "type": "number"
            },
            full_text="a"
        )

        llm.get_logits_from_input_ids.side_effect = \
            lambda ids: [100.0] + [0.0] * 255
        llm.decode.side_effect = lambda ids: "" if ids == [0] else "x"

        result = gen.generate(
            make_prompt("no numbers here"), func_def, "a", "number", []
        )
        assert result is None

    def test_get_valid_num_extracts_integers(self):
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        assert gen.get_valid_num("Add 3 and 42") == ["3", "42"]

    def test_get_valid_num_extracts_floats(self):
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        assert gen.get_valid_num("value is 3.14") == ["3.14"]

    def test_get_valid_num_extracts_negatives(self):
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        assert gen.get_valid_num("temperature is -5") == ["-5"]

    def test_get_valid_num_returns_empty_for_no_numbers(self):
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        assert gen.get_valid_num("no numbers here") == []

    def test_skips_used_candidates(self):
        """When a number is already used, it should not be reused."""
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=100)
        func_def = make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
                },
            returns={
                "type": "number"
            },
            full_text="a"
        )
        prompt = make_prompt("Add 3 and 7")

        # Make model generate tokens for "7" (second number)
        target_ids = gen.encode_cache("7")
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(target_ids):
                logits[target_ids[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)

        # "3" is already used — should get "7"
        result = gen.generate(prompt, func_def, "b", "number", ["3"])
        assert float(result) == 7.0

    def test_falls_back_to_free_generation_when_no_arabic_numbers(self):
        """
        When no arabic numbers in prompt, should fall back to free generation.
        _generate_free is called by checking that the model generates freely.
        """
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        func_def = make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
                },
            returns={
                "type": "number"
            },
            full_text="a"
        )
        # Prompt has no arabic numbers — free generation path
        prompt = make_prompt("Add forty-two and seven")

        # Model generates "4", "2" to form "42"
        tokens = [ord("4"), ord("2")]
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(tokens):
                logits[tokens[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0  # EOS
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)

        result = gen.generate(prompt, func_def, "a", "number", [])
        assert float(result) == 42.0

    def test_extracts_negative_number_in_free_generation(self):
        """
        Negative numbers should be extracted correctly in free generation too.
        """
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        func_def = make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
                },
            returns={
                "type": "number"
            },
            full_text="a"
        )
        # Prompt has no arabic numbers — free generation path
        prompt = make_prompt("Add -5 and seven")

        # Model generates "-", "5" to form "-5"
        tokens = [ord("a"), ord("-"), ord("5"), ord("-")]
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(tokens):
                logits[tokens[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0  # EOS
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)

        result = gen.generate(prompt, func_def, "a", "number", [])
        assert result == "-5"

    def test_rejects_lone_minue_in_free_generation(self):
        """
        Lone minus signe should be rejected in free generation too.
        """
        llm = make_mock_llm()
        gen = NumParamGenerator(llm, max_tokens=10)
        func_def = make_func_def(
            name="fn_add_numbers",
            description="Add two numbers together and return their sum.",
            parameters={
                "a": {
                    "type": "number"
                },
                "b": {
                    "type": "number"
                }
                },
            returns={
                "type": "number"
            },
            full_text="a"
        )
        # Prompt has no arabic numbers — free generation path
        prompt = make_prompt("Add -5 ans seven")

        # Model generates single "-"
        tokens = [ord("a"), ord("-"), ord("a"), ord("-")]
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(tokens):
                logits[tokens[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0  # EOS
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)

        result = gen.generate(prompt, func_def, "a", "number", [])
        assert result is None


# ------------------------------------------------------------------ #
# BoolParamGenerator tests                                          #
# ------------------------------------------------------------------ #

class TestBoolParamGenerator:

    def _make_gen_that_outputs(self, target: str) -> BoolParamGenerator:
        """Helper that wires the mock LLM to output the given target string."""
        llm = make_mock_llm(vocab_size=256)
        gen = BoolParamGenerator(llm, max_tokens=100)
        target_ids = gen.encode_cache(target)
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(target_ids):
                logits[target_ids[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)
        return gen

    def test_returns_true_for_true(self):
        gen = self._make_gen_that_outputs("true")
        func_def = make_func_def(
            name="fn_check",
            description="Check whether it is true or not.",
            parameters={"flag": {"type": "boolean"}},
            returns={},
            full_text="a"
        )
        result = gen.generate(
            make_prompt("Is it true?"), func_def, "flag", "boolean"
        )
        assert result is True

    def test_returns_false_for_false(self):
        gen = self._make_gen_that_outputs("false")
        func_def = make_func_def(
            name="fn_check",
            description="Check whether it is true or not.",
            parameters={"flag": {"type": "boolean"}},
            returns={},
            full_text="a"
        )
        result = gen.generate(
            make_prompt("Is it false?"), func_def, "flag", "boolean"
        )
        assert result is False

    def test_returns_none_on_eos(self):
        llm = make_mock_llm(vocab_size=256)
        gen = BoolParamGenerator(llm, max_tokens=10)
        func_def = make_func_def(
            name="fn_check",
            description="Check whether it is true or not.",
            parameters={"flag": {"type": "boolean"}},
            returns={},
            full_text="a"
        )
        llm.get_logits_from_input_ids.side_effect = \
            lambda ids: [100.0] + [0.0] * 255
        llm.decode.side_effect = lambda ids: ""

        result = gen.generate(
            make_prompt("check"), func_def, "flag", "boolean"
        )
        assert result is None


# ------------------------------------------------------------------ #
# StrParamGenerator tests                                           #
# ------------------------------------------------------------------ #

class TestStrParamGenerator:

    def test_extract_string_finds_double_quoted(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_string('Replace "hello" with "world"')
        assert "hello" in result
        assert "world" in result

    def test_extract_string_finds_single_quoted(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_string("Reverse 'Python'")
        assert "Python" in result

    def test_extract_string_returns_empty_for_no_quotes(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_string("no quoted strings here")
        assert result == []

    def test_extract_replacement_semantic_asterisk(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_replacement_candidates("Replace vowels with asterisks")
        assert "*" in result

    def test_extract_replacement_semantic_empty(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_replacement_candidates("Remove all spaces with nothing")
        assert "" in result

    def test_extract_replacement_quoted(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_replacement_candidates("Replace 'cat' with 'dog'")
        assert "dog" in result

    def test_extract_replacement_uppercase_literal(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_replacement_candidates(
            'Replace numbers with PLACEHOLDER in the text'
        )
        assert "PLACEHOLDER" in result

    def test_extract_regex_candidates_digits(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_regex_candidates("replace all digits")
        assert r"\d+" in result

    def test_extract_regex_candidates_vowels(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_regex_candidates("replace all vowels")
        assert r"[aeiouAEIOU]" in result

    def test_extract_regex_candidates_lowercase_vowels(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_regex_candidates(
            "replace all lowercase vowels"
        )
        assert r"[aeiou]" in result
        assert r"[aeiouAEIOU]" not in result

    def test_extract_regex_candidates_uppercase_letters(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_regex_candidates(
            "replace all uppercase letters"
        )
        assert r"[A-Z]" in result
        assert r"[a-zA-Z]" not in result

    def test_extract_regex_candidates_empty_for_unknown(self):
        llm = make_mock_llm()
        gen = StrParamGenerator(llm, max_tokens=100)
        result = gen.extract_regex_candidates(
            "replace something unrecognised"
        )
        assert result == []

    def test_source_string_prefers_longest(self):
        """
        source_string parameter should receive the longest quoted string.
        """
        llm = make_mock_llm(vocab_size=256)
        gen = StrParamGenerator(llm, max_tokens=100)
        func_def = make_func_def(
            "fn_substitute_string_with_regex",
            description="Replace all occurrences matching a regex pattern in a string.",
            parameters={
                "source_string": {
                    "type": "string"
                },
                "regex": {"type": "string"},
                "replacement": {
                    "type": "string"
                },
            },
            returns={"type": "string"},
            full_text="a"
        )
        prompt = make_prompt(
            "Substitute 'cat' with 'dog' in "
            "'The cat sat on the mat with another cat'"
        )

        result = gen.generate(
            prompt, func_def, "source_string", "string", []
        )
        assert result == "The cat sat on the mat with another cat"

    def test_name_param_uses_free_generation(self):
        """name parameter should always use free generation."""
        llm = make_mock_llm(vocab_size=256)
        gen = StrParamGenerator(llm, max_tokens=10)
        func_def = make_func_def(
            "fn_greet",
            description="Generate a greeting message for a person by name.",
            parameters={
                "name": {"type": "string"}
            },
            returns={"type": "string"},
            full_text="a"
        )

        # Generate tokens for "Maria"
        tokens = [ord(c) for c in "Maria"]
        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            if call_count[0] < len(tokens):
                logits[tokens[call_count[0]]] = 100.0
            else:
                logits[0] = 100.0
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else \
            "".join(chr(i) for i in ids if 0 < i < 128)

        result = gen.generate(
            make_prompt("Greet Maria"), func_def, "name", "string", []
        )
        assert result == "Maria"


# ------------------------------------------------------------------ #
# ParameterGenerator integration tests                                 #
# ------------------------------------------------------------------ #

class TestParameterGenerator:

    def test_returns_empty_dict_for_no_parameters(self):
        llm = make_mock_llm()
        gen = ParameterGenerator(llm, max_tokens=50)
        func_def = make_func_def("fn_no_params", description="a", parameters={}, returns={}, full_text="a")
        result = gen.generate(make_prompt("do something"), func_def)
        assert result == {}

    def test_dispatches_number_to_numeric_generator(self):
        """generate() should call num_gen.generate for number params."""
        llm = make_mock_llm()
        gen = ParameterGenerator(llm, max_tokens=50)
        func_def = make_func_def(
            "fn_add",
            description="add a number to another existing number",
            parameters={"a": {"type": "number"}},
            returns={"type": "number"},
            full_text="a"
        )
        gen.num_param.generate = MagicMock(return_value=42.0)

        result = gen.generate(make_prompt("add 42"), func_def)

        gen.num_param.generate.assert_called_once()
        assert result["a"] == 42.0

    def test_dispatches_boolean_to_boolean_generator(self):
        """generate() should call bool_gen.generate for boolean params."""
        llm = make_mock_llm()
        gen = ParameterGenerator(llm, max_tokens=50)
        func_def = make_func_def(
            "fn_check",
            description="a",
            parameters={"flag": {"type": "boolean"}},
            returns={"type": "boolean"},
            full_text="a"
        )
        gen.bool_param.generate = MagicMock(return_value=True)

        result = gen.generate(make_prompt("check true"), func_def)

        gen.bool_param.generate.assert_called_once()
        assert result["flag"] is True

    def test_dispatches_string_to_string_generator(self):
        """generate() should call str_gen.generate for string params."""
        llm = make_mock_llm()
        gen = ParameterGenerator(llm, max_tokens=50)
        func_def = make_func_def(
            "fn_greet",
            description="Generate a greeting message for a person by name.",
            parameters={
                "name": {"type": "string"}
            },
            returns={"type": "string"},
            full_text="a"
        )
        gen.str_param.generate = MagicMock(return_value="Maria")

        result = gen.generate(make_prompt("greet Maria"), func_def)

        gen.str_param.generate.assert_called_once()
        assert result["name"] == "Maria"

    def test_tracks_used_candidates_across_parameters(self):
        """
        Used candidates from earlier parameters should be passed to
        later ones to avoid reuse.
        """
        llm = make_mock_llm()
        gen = ParameterGenerator(llm, max_tokens=50)
        func_def = make_func_def(
            "fn_substitute",
            description="Replace all occurrences matching a regex pattern in a string.",
            parameters={
                "source_string": {
                    "type": "string"
                },
                "regex": {"type": "string"},
                "replacement": {
                    "type": "string"
                },
            },
            returns={"type": "string"},
            full_text="a"
        )

        call_args: list[list[str]] = []

        def mock_str_generate(prompt, func_def, var_name,
                               param_type, used_can):
            call_args.append(used_can.copy())
            return f"value_{var_name}"

        gen.str_param.generate = MagicMock(side_effect=mock_str_generate)

        gen.generate(make_prompt("test"), func_def)

        # First call should have empty used_can
        assert call_args[0] == []
        # Second call should include the first parameter's value
        assert "value_source_string" in call_args[1]

    def test_none_values_not_added_to_used_candidates(self):
        """None results should not be added to used_candidates."""
        llm = make_mock_llm()
        gen = ParameterGenerator(llm, max_tokens=50)
        func_def = make_func_def(
            "fn_test",
            description="a",
            parameters={
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            returns={},
            full_text="a"
        )

        call_args: list[list[str]] = []

        def mock_str_generate(prompt, func_def, var_name,
                               param_type, used_can):
            call_args.append(used_can.copy())
            return None  # first param returns None

        gen.str_param.generate = MagicMock(side_effect=mock_str_generate)
        gen.generate(make_prompt("test"), func_def)

        # Second call should still have empty used_can since first was None
        assert call_args[1] == []
