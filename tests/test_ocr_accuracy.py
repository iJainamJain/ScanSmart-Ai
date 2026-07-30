import numpy as np
import pytest

from src.enhancement.illumination import (
    estimate_illumination,
    flatten_illumination,
    illumination_unevenness,
)
from src.evaluation.ocr_accuracy import (
    VariantResult,
    character_error_rate,
    levenshtein,
    normalize_text,
    summarise_variants,
    word_error_rate,
)
from src.evaluation.synthetic import (
    add_cast_shadow,
    add_illumination_gradient,
    add_ruled_lines,
    build_benchmark_page,
    render_text_page,
)


# ----------------------------------------------------------------- edit distance
def test_levenshtein_counts_single_edits():
    assert levenshtein("kitten", "kitten") == 0
    assert levenshtein("kitten", "sitten") == 1   # substitution
    assert levenshtein("kitten", "kitte") == 1    # deletion
    assert levenshtein("kitten", "kittens") == 1  # insertion


def test_levenshtein_is_symmetric():
    assert levenshtein("abcdef", "azced") == levenshtein("azced", "abcdef")


def test_cer_is_zero_for_an_exact_read():
    assert character_error_rate("hello world", "hello world") == 0.0


def test_cer_ignores_whitespace_and_case_differences():
    """OCR line-wrapping varies run to run; that must not count as errors."""
    assert character_error_rate("Hello World", "hello\n  world  ") == 0.0


def test_cer_scales_with_the_number_of_wrong_characters():
    assert character_error_rate("abcd", "abcX") == pytest.approx(0.25)


def test_cer_rejects_an_empty_reference():
    with pytest.raises(ValueError):
        character_error_rate("", "anything")


def test_wer_counts_whole_words_not_characters():
    # One wrong word out of four: 0.25 by word, far less by character.
    assert word_error_rate("the cat sat down", "the dog sat down") == pytest.approx(0.25)


def test_normalize_collapses_runs_of_whitespace():
    assert normalize_text("a \n\t b") == "a b"


# --------------------------------------------------------------------- reporting
def test_summary_marks_a_lower_cer_as_better():
    results = [
        VariantResult("baseline", 0.50, 0.6, 100),
        VariantResult("flattened", 0.30, 0.4, 100),
    ]
    text = summarise_variants(results)
    assert "better" in text and "-0.200" in text


def test_summary_marks_a_higher_cer_as_worse():
    results = [
        VariantResult("baseline", 0.30, 0.4, 100),
        VariantResult("flattened", 0.50, 0.6, 40),
    ]
    assert "worse" in summarise_variants(results)


# ------------------------------------------------------------------ illumination
def test_flattening_evens_out_a_lighting_gradient():
    page, _ = render_text_page()
    lit = add_illumination_gradient(page, strength=0.6)

    before = illumination_unevenness(lit)
    flattened = flatten_illumination(lit)

    assert before > 5.0, "the synthetic gradient should register as uneven"
    # Measured on the estimated field of the result, which the division flattens.
    assert illumination_unevenness(flattened) < before


def test_unevenness_is_near_zero_for_a_uniformly_lit_page():
    page, _ = render_text_page()
    assert illumination_unevenness(page) < 2.0


def test_unevenness_returns_a_number_for_a_perfectly_flat_image():
    """A uniform field once produced nan via an empty percentile slice."""
    value = illumination_unevenness(np.full((200, 200), 200, np.uint8))
    assert not np.isnan(value)


def test_estimate_illumination_removes_text_but_keeps_the_paper_level():
    page, _ = render_text_page()
    field = estimate_illumination(page)
    # Text pixels are dark in the original and paper-coloured in the field.
    assert page.min() < 100
    assert field.min() > 150


def test_estimate_illumination_rejects_a_colour_image():
    with pytest.raises(ValueError):
        estimate_illumination(np.zeros((50, 50, 3), np.uint8))


def test_flattening_leaves_an_already_even_page_roughly_alone():
    page, _ = render_text_page()
    flattened = flatten_illumination(page)
    # Allowed to rescale brightness, but must not destroy the text contrast.
    assert flattened.std() > page.std() * 0.5


# --------------------------------------------------------------------- synthetic
def test_benchmark_page_returns_matching_text():
    page, truth = build_benchmark_page()
    assert page.ndim == 2
    assert "quick brown fox" in truth


def test_each_degradation_can_be_toggled_off():
    plain, _ = build_benchmark_page(ruled=False, gradient=False, shadow=False, noise=False)
    ruled, _ = build_benchmark_page(ruled=True, gradient=False, shadow=False, noise=False)
    assert not np.array_equal(plain, ruled)


def test_ruling_is_fainter_than_the_ink():
    """Real notebook ruling is much lighter than handwriting - which is why
    illumination flattening can remove one and keep the other."""
    page, _ = render_text_page()
    ruled = add_ruled_lines(page)
    added = ruled[ruled != page]
    assert added.size > 0
    assert added.min() > page.min(), "ruling must be lighter than the darkest ink"


def test_cast_shadow_darkens_part_of_the_page_only():
    page, _ = render_text_page()
    shadowed = add_cast_shadow(page, strength=0.5)
    assert shadowed.mean() < page.mean()
    assert shadowed.max() >= page.max() - 5, "some of the page stays unshadowed"
