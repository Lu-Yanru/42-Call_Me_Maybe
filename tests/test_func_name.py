import pytest
from unittest.mock import MagicMock, patch
from src.generator_funcname import FuncNameGenerator
from tests.conftest import make_func_def, make_prompt, make_mock_llm


class TestFuncNameGeneratorInit:
    """Tests for FuncNameGenerator initialisation."""

    def test_init_stores_funcs(self, mock_llm, sample_funcs):
        gen = FuncNameGenerator(mock_llm, max_tokens=50,
                                    funcs=sample_funcs)
        assert gen.funcs == sample_funcs

    def test_init_stores_max_tokens(self, mock_llm, sample_funcs):
        gen = FuncNameGenerator(mock_llm, max_tokens=99,
                                    funcs=sample_funcs)
        assert gen.max_tokens == 99

    def test_init_empty_funcs(self, mock_llm):
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=[])
        assert gen.funcs == []


class TestFuncNameGeneratorGenerate:
    """Tests for FuncNameGenerator.generate()."""

    def test_returns_matched_function_name(self, sample_funcs):
        """
        When the model generates tokens that spell out a function name,
        generate() should return that name.
        """
        llm = make_mock_llm()
        gen = FuncNameGenerator(llm, max_tokens=100, funcs=sample_funcs)

        # Make the model generate tokens that spell "fn_greet"
        target = "fn_greet"
        token_ids = gen.encode_cache(target)

        call_count = 0

        def mock_logits(input_ids: list[int]) -> list[float]:
            nonlocal call_count
            logits = [0.0] * 256
            # Return tokens of target one by one, then EOS
            if call_count < len(token_ids):
                logits[token_ids[call_count]] = 100.0
            else:
                # Signal EOS via a token that decodes to ""
                logits[0] = 100.0
            call_count += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        # Make token 0 decode to "" so it is recognised as EOS
        original_decode = llm.decode.side_effect

        def mock_decode(ids):
            if ids == [0]:
                return ""
            return original_decode(ids)

        llm.decode.side_effect = mock_decode

        result = gen.generate(make_prompt("greet Maria"))
        assert result == target

    def test_returns_none_on_eos_before_match(self, sample_funcs):
        """
        When the model generates EOS before completing any function name,
        generate() should return None.
        """
        llm = make_mock_llm()
        gen = FuncNameGenerator(llm, max_tokens=100, funcs=sample_funcs)

        # Always return EOS
        def mock_logits(input_ids):
            logits = [0.0] * 256
            logits[0] = 100.0  # token 0 is EOS
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits
        llm.decode.side_effect = lambda ids: "" if ids == [0] else "x"

        result = gen.generate(make_prompt("greet Maria"))
        assert result is None

    def test_returns_none_when_max_tokens_reached(self, sample_funcs):
        """
        When max_tokens is reached without a match, return None.
        """
        llm = make_mock_llm()
        # Very low max_tokens so we run out before matching
        gen = FuncNameGenerator(llm, max_tokens=2, funcs=sample_funcs)

        call_count = [0]

        def mock_logits(input_ids):
            logits = [0.0] * 256
            # Generate harmless tokens that never match any function name
            logits[ord("x")] = 100.0
            call_count[0] += 1
            return logits

        llm.get_logits_from_input_ids.side_effect = mock_logits

        result = gen.generate(make_prompt("do something"))
        assert result is None
        # Should have stopped at max_tokens
        assert call_count[0] <= 2

    def test_encode_cache_is_called(self, mock_llm, sample_funcs):
        """encode_cache should be called when building the prompt."""
        gen = FuncNameGenerator(mock_llm, max_tokens=5, funcs=sample_funcs)
        mock_llm.get_logits_from_input_ids.side_effect = \
            lambda ids: [0.0] * 256

        gen.generate(make_prompt("test"))
        assert mock_llm.encode.called


class TestFuncNameGeneratorTokenization:
    """Tests for tokenization helpers."""

    def test_tokenize_str_returns_tuples(self, mock_llm, sample_funcs):
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        result = gen.tokenize_str(["hello"])
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_tokenize_str_no_duplicates(self, mock_llm, sample_funcs):
        """tokenize_str should deduplicate identical token sequences."""
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        result = gen.tokenize_str(["ab"])
        seen = set()
        for ids, _ in result:
            key = tuple(ids)
            assert key not in seen, f"Duplicate token sequence: {ids}"
            seen.add(key)

    def test_tokenize_str_empty_input(self, mock_llm, sample_funcs):
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        result = gen.tokenize_str([])
        assert result == []

    def test_get_valid_next_returns_none_when_no_active(
            self, mock_llm, sample_funcs):
        """get_valid_next should return None when no candidate is active."""
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        candidates = [([1, 2, 3], "abc")]
        match_progress = [None]
        result = gen.get_valid_next(candidates, match_progress)
        assert result is None

    def test_get_valid_next_returns_set_when_active(
            self, mock_llm, sample_funcs):
        """get_valid_next should return the valid next token ids."""
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        candidates = [([1, 2, 3], "abc"), ([4, 5], "de")]
        match_progress = [1, 0]  # first is at position 1, second at 0
        result = gen.get_valid_next(candidates, match_progress)
        assert result == {2, 4}

    def test_update_match_progress_activates_candidate(
            self, mock_llm, sample_funcs):
        """update_match_progress should activate a candidate on first token."""
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        candidates = [([10, 20], "ab")]
        match_progress = [None]
        result = gen.update_match_progress(10, candidates, match_progress)
        assert result is None
        assert match_progress[0] == 1

    def test_update_match_progress_returns_on_full_match(
            self, mock_llm, sample_funcs):
        """update_match_progress should return the name on full match."""
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        candidates = [([10, 20], "ab")]
        match_progress = [1]  # already matched first token
        result = gen.update_match_progress(20, candidates, match_progress)
        assert result == "ab"

    def test_update_match_progress_resets_on_mismatch(
            self, mock_llm, sample_funcs):
        """update_match_progress should deactivate a candidate on mismatch."""
        gen = FuncNameGenerator(mock_llm, max_tokens=50, funcs=sample_funcs)
        candidates = [([10, 20], "ab")]
        match_progress = [1]
        result = gen.update_match_progress(99, candidates, match_progress)
        assert result is None
        assert match_progress[0] is None
