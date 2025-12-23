from treesearch.llm.query import prompt_to_md


def test_string_only():
    assert prompt_to_md("text") == "text"


def test_simple_header_with_text():
    assert prompt_to_md({"h": "text"}) == "# h\ntext"


def test_header_without_text():
    assert prompt_to_md({"h": {}}) == "# h"


def test_nested_headers_with_text():
    assert prompt_to_md({"h1": {"h2": "text"}}) == "# h1\n## h2\ntext"


def test_list_of_text():
    assert prompt_to_md(["a", "b"]) == "a\nb"


def test_text_then_header_in_list_adds_blank_line():
    assert prompt_to_md({"h": ["text", {"sub": "more"}]}) == (
        "# h\ntext\n\n## sub\nmore"
    )


def test_header_then_text_in_list_no_blank_line():
    assert prompt_to_md({"h": [{"sub": "more"}, "text"]}) == ("# h\n## sub\nmore\ntext")


def test_multiple_text_entries():
    assert prompt_to_md({"h": ["a", "b"]}) == "# h\na\nb"


def test_deeply_nested_mixed_structure():
    assert (
        prompt_to_md({"h": ["a", {"h2": ["b", {"h3": "c"}]}]})
        == "# h\na\n\n## h2\nb\n\n### h3\nc"
    )


def test_empty_string():
    assert prompt_to_md("") == ""


def test_empty_list():
    assert prompt_to_md([]) == ""


def test_list_with_only_empty_values():
    assert prompt_to_md(["", {}, []]) == ""
