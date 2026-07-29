from continual_pt.learn import GroundedExampleBuilder
from continual_pt.schema import SourceDocument


def test_practice_curriculum_interleaves_sources_before_reusing_one():
    first = SourceDocument(
        url="https://example.com/first",
        title="first",
        text="",
        sections=[("first-a", "a" * 100), ("first-b", "b" * 100)],
    )
    second = SourceDocument(
        url="https://example.com/second",
        title="second",
        text="",
        sections=[("second-a", "c" * 100), ("second-b", "d" * 100)],
    )

    curriculum = GroundedExampleBuilder._balanced_sections([first, second])

    assert [entry[1] for entry in curriculum] == ["first-a", "second-a", "first-b", "second-b"]
