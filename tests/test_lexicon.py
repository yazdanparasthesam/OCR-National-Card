"""The name lexicon must *help* and never invent a different name."""

from __future__ import annotations

import pytest

from nidocr.fields.lexicon import NameLexicon, get_lexicon


@pytest.fixture(scope="module")
def lexicon() -> NameLexicon:
    """A tiny hand-written lexicon so the snapping rules can be asserted exactly."""
    return NameLexicon(
        first_names=["علی", "محمد", "رضا", "نازنین", "آرمین", "یاسر"],
        last_names=["محمدی", "رضایی", "بیات", "تبریزی", "پارسایی"],
    )


def test_exact_membership(lexicon):
    """Exact membership works per kind and rejects unknown names."""
    assert lexicon.contains("رضایی", "last")
    assert lexicon.contains("آرمین", "first")
    assert not lexicon.contains("کاملا", "first")


def test_snaps_a_confused_dot(lexicon):
    """A dotted-letter OCR error is snapped to the real name."""
    assert lexicon.snap("رضابی", "last")[0] == "رضایی"
    assert lexicon.snap("پیات", "last")[0] == "بیات"


def test_folds_alef_variants(lexicon):
    """«ا» and «آ» are folded for lookup while the canonical spelling is returned."""
    assert lexicon.contains("ارمین", "first")
    # an exact (folded) hit is not a "near miss": the canonical spelling is used
    assert lexicon.snap("ارمین", "first")[0] is None
    assert lexicon.canonical("ارمین", "first") == "آرمین"


def test_never_adds_or_drops_letters(lexicon):
    # "محمدی" -> "محمد" would drop a letter: not a re-dotting error
    """A different name length is never «repaired» into another name."""
    assert lexicon.snap("محمد", "last")[0] is None
    assert lexicon.snap("علیی", "first")[0] is None


def test_unknown_name_stays_untouched(lexicon):
    """A name that is not near any lexicon entry is returned as-is."""
    assert lexicon.snap("کامران", "first")[0] is None


def test_missing_lexicon_is_a_no_op():
    """Without a lexicon the matcher silently disables itself."""
    empty = NameLexicon()
    assert not empty.available("first")
    assert empty.snap("رضابی", "last") == (None, 0.0)


def test_shipped_lexicon_loads():
    """The bundled lexicon files load and are not empty."""
    shipped = get_lexicon()
    assert shipped.size("first") > 1000
    assert shipped.size("last") > 1000
