"""Regression tests for both deliverables; all API calls are mocked."""

from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest

from test_template import SOLUTION_PATH, TEMPLATE_PATH, _load_module


@pytest.fixture(params=[TEMPLATE_PATH, SOLUTION_PATH], ids=["template", "solution"])
def implementation(request):
    return _load_module(request.param, f"edge_{request.param.stem}")


def test_gemini_missing_usage(implementation):
    with patch("google.genai.Client") as client:
        client.return_value.models.generate_content.return_value = SimpleNamespace(
            text=None, usage_metadata=None
        )
        text, latency, usage = implementation.call_gemini("Hello")
    assert text == ""
    assert latency >= 0
    assert usage == {"input_tokens": 0, "output_tokens": 0}


def test_streaming_keeps_three_complete_previous_turns(implementation, capsys):
    prompts = [f"question {index}" for index in range(5)]
    with patch("google.genai.Client") as client, patch(
        "builtins.input", side_effect=["", *prompts, "EXIT"]
    ):
        stream = client.return_value.models.generate_content_stream
        stream.side_effect = [
            [SimpleNamespace(text="Hello"), SimpleNamespace(text=None),
             SimpleNamespace(text=" world")]
            for _ in prompts
        ]
        implementation.streaming_chatbot()

    assert stream.call_count == 5
    for index, request in enumerate(stream.call_args_list):
        messages = request.kwargs["contents"]
        assert len(messages) == min(index, 3) * 2 + 1
        assert [message["role"] for message in messages] == (
            ["user", "model"] * min(index, 3) + ["user"]
        )
        assert messages[0]["parts"][0]["text"] == prompts[max(0, index - 3)]
        assert messages[-1]["parts"][0]["text"] == prompts[index]
        for message in messages[1::2]:
            assert message["parts"][0]["text"] == "Hello world"
    assert "Gemini: Hello world" in capsys.readouterr().out


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_chatbot_handles_terminal_exit(implementation, exception):
    with patch("google.genai.Client") as client, patch(
        "builtins.input", side_effect=exception
    ):
        implementation.streaming_chatbot()
        client.return_value.models.generate_content_stream.assert_not_called()


def test_retry_schedule_and_original_exception(implementation):
    error = RuntimeError("unavailable")
    operation = Mock(side_effect=error)
    with patch.object(implementation.time, "sleep") as sleep:
        with pytest.raises(RuntimeError) as raised:
            implementation.retry_with_backoff(operation, max_retries=3, base_delay=0.25)
    assert raised.value is error
    assert operation.call_count == 4
    assert sleep.call_args_list == [call(0.25), call(0.5), call(1.0)]


def test_markdown_truncates_before_escaping(implementation):
    stats = {
        "response": "a" * 46 + "|" + "b" * 20 + "\r\nnext line",
        "latency": 0.5, "cost": 0.00045,
        "input_tokens": 10, "output_tokens": 20,
    }
    result = {"prompt": "first|second\r\nthird"}
    result.update(dict.fromkeys(["gpt4o", "gpt4o_mini", "gemini_flash"], stats))
    table = implementation.format_comparison_table([result])
    assert len(table.splitlines()) == 5
    assert "first\\|second third" in table
    assert "a" * 46 + "\\|..." in table
    assert "$0.00045000" in table
    assert len(implementation.format_comparison_table([]).splitlines()) == 2


def test_empty_batch_does_not_call_models(implementation):
    with patch.object(implementation, "compare_models") as compare:
        assert implementation.batch_compare([]) == []
        compare.assert_not_called()
