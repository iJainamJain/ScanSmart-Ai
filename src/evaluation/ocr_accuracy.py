"""OCR accuracy measurement: character and word error rate against known text.

This exists because "the output looks cleaner" is not evidence. Several
image-quality metrics invented for this project were discarded after failing
to agree with human judgement, so enhancement claims are measured here
instead - against ground-truth text, which cannot be argued with.

It matters most for illumination flattening, which is subtractive: it removes
ink rather than recovering it, so a visibly cleaner page and a page with the
faint writing destroyed look very similar. CER tells them apart.
"""

import re
from dataclasses import dataclass

import numpy as np
import pytesseract

from src.ocr.config import tesseract_available  # noqa: F401  (re-exported)


def levenshtein(reference: str, hypothesis: str) -> int:
    """Edit distance, computed in O(min(len)) space."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ref_char != hyp_char),  # substitution
                )
            )
        previous = current
    return previous[-1]


def normalize_text(text: str, fold_case: bool = True) -> str:
    """Collapse whitespace (and optionally case) so formatting noise isn't scored.

    OCR line-wrapping differs run to run; without this, a correct read that
    breaks lines differently would be penalised as errors.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if fold_case else text


def character_error_rate(reference: str, hypothesis: str, fold_case: bool = True) -> float:
    """Edit distance per reference character. 0.0 is perfect; >1.0 is possible."""
    reference = normalize_text(reference, fold_case)
    hypothesis = normalize_text(hypothesis, fold_case)
    if not reference:
        raise ValueError("reference text is empty; CER is undefined")
    return levenshtein(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str, fold_case: bool = True) -> float:
    """Edit distance over words rather than characters."""
    ref_words = normalize_text(reference, fold_case).split()
    hyp_words = normalize_text(hypothesis, fold_case).split()
    if not ref_words:
        raise ValueError("reference text is empty; WER is undefined")

    # Reuse the character routine by mapping each distinct word to one symbol.
    vocabulary = {w: chr(i + 256) for i, w in enumerate({*ref_words, *hyp_words})}
    return levenshtein(
        "".join(vocabulary[w] for w in ref_words),
        "".join(vocabulary[w] for w in hyp_words),
    ) / len(ref_words)


def ocr_text(image: np.ndarray) -> str:
    """Read text from an image with Tesseract; returns "" if it is unavailable."""
    try:
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


def tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@dataclass
class VariantResult:
    variant: str
    cer: float
    wer: float
    characters_read: int


def compare_variants(variants: dict[str, np.ndarray], reference_text: str) -> list[VariantResult]:
    """OCR each pipeline variant of the same page and score it against the truth.

    `variants` maps a name ("baseline", "flattened", ...) to the image that
    variant produces, so any enhancement change can be scored the same way.
    """
    results = []
    for name, image in variants.items():
        hypothesis = ocr_text(image)
        results.append(
            VariantResult(
                variant=name,
                cer=character_error_rate(reference_text, hypothesis),
                wer=word_error_rate(reference_text, hypothesis),
                characters_read=len(normalize_text(hypothesis)),
            )
        )
    return results


def summarise_variants(results: list[VariantResult], baseline: str = "baseline") -> str:
    """Report each variant's CER and its change against the baseline."""
    if not results:
        return "No variants evaluated."

    by_name = {r.variant: r for r in results}
    lines = [f"{'variant':<16}{'CER':>8}{'WER':>8}{'chars':>8}{'vs base':>10}"]
    reference = by_name.get(baseline)
    for result in results:
        if reference is not None and result.variant != baseline:
            delta = result.cer - reference.cer
            change = f"{delta:+.3f}" + (" better" if delta < 0 else " worse" if delta > 0 else "")
        else:
            change = "-"
        lines.append(
            f"{result.variant:<16}{result.cer:8.3f}{result.wer:8.3f}"
            f"{result.characters_read:8d}{change:>10}"
        )
    return "\n".join(lines)
