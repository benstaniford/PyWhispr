from pywhispr.history import HISTORY_SIZE, PREVIEW_CHARS, TranscriptHistory, preview


def test_newest_first_and_bounded():
    history = TranscriptHistory(size=3)
    for text in ("one", "two", "three", "four"):
        history.remember(text)
    assert list(history) == ["four", "three", "two"]
    assert len(history) == 3


def test_default_size_is_the_constant():
    history = TranscriptHistory()
    for i in range(HISTORY_SIZE + 2):
        history.remember(f"line {i}")
    assert len(history) == HISTORY_SIZE


def test_blank_transcripts_are_not_kept():
    history = TranscriptHistory()
    history.remember("")
    history.remember("   \n ")
    assert list(history) == []


def test_an_immediate_repeat_is_one_entry():
    # Dictating the same thing twice (the second time because the first went
    # nowhere) should not fill the picker with duplicates.
    history = TranscriptHistory()
    history.remember("hello there")
    history.remember("hello there")
    assert list(history) == ["hello there"]
    # Only *immediate* repeats: the same words again later are worth keeping.
    history.remember("something else")
    history.remember("hello there")
    assert list(history) == ["hello there", "something else", "hello there"]


def test_clear_empties_it():
    history = TranscriptHistory()
    history.remember("hello")
    history.clear()
    assert list(history) == []


def test_preview_is_one_line():
    assert preview("hello   there\nworld") == "hello there world"


def test_preview_is_clipped_to_width():
    assert preview("x" * 100, width=10) == "x" * 9 + "…"
    assert len(preview("y" * 200)) == PREVIEW_CHARS
    assert preview("short", width=10) == "short"
