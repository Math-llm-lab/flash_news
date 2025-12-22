from get_news import parse_llm_output


def test_parse_llm_output_handles_messy_text_case_and_extra_lines():
    text = "Historical: False\nSentiment: Significant Rise\nExtra notes..."
    is_hist, sent = parse_llm_output(text)
    assert is_hist is False
    assert sent == "significant rise"


def test_parse_llm_output_returns_none_when_missing_fields():
    is_hist, sent = parse_llm_output("no booleans here")
    assert is_hist is None
    assert sent is None
