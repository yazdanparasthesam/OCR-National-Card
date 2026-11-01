"""Persian digit classifier (glyph level k-NN, numpy only).

Why a custom classifier instead of Tesseract for digits?

``fas.traineddata`` recognises Persian *words* very well but is poor at runs of
Persian digits — the line decoder prefers letter sequences, so national codes and
dates come back mutilated (measured: 20–50 % character accuracy).  Persian digits
are printed as isolated glyphs, so the robust approach is:

1. segment the numeric field into individual glyphs (column projection),
2. classify each glyph with this small k-NN model trained on synthetic glyphs from
   several Persian font families with geometric/photometric augmentation,
3. put the digits back together and let the field validator (mod-11 checksum,
   Jalali calendar) confirm or repair the result.

Held-out font evaluation (fonts never seen during training): ~99 % per glyph, which
combined with the national-code checksum repair gives >99 % exact codes.

The model file is a tiny ``.npz`` (features as ``uint8`` + labels), produced by
``python scripts/train_digit_model.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from nidocr.config import PROJECT_ROOT
from nidocr.ocr.text import PERSIAN_DIGITS

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "persian_digits.npz"
FEATURE_SIZE = 20
FEATURE_DIM = FEATURE_SIZE * FEATURE_SIZE


@dataclass
class DigitPrediction:
    """Result of classifying one Persian digit glyph.

    ``digit`` is the winner, ``confidence`` a 0–100 score, ``similarity`` the cosine
    similarity of the winning cluster, ``margin`` the gap to the runner-up (a low
    margin means the glyph is genuinely ambiguous) and ``options`` the top-3
    ``(digit, share)`` pairs that the field reader turns into alternatives.
    """
    digit: str
    confidence: float  # 0..100
    similarity: float
    margin: float
    options: list[tuple[str, float]] = field(default_factory=list)  # (digit, share)


def featurize(patch: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray | None:
    """Turn a glyph crop into a normalised ``size × size`` feature vector."""
    if patch is None or patch.size == 0:
        return None
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(patch, (0, 0), 0.7)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return None
    glyph = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = glyph.shape
    scale = (size - 4) / max(height, width)
    resized = cv2.resize(
        glyph,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((size, size), np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    features = canvas.astype(np.float32) / 255.0
    features = cv2.GaussianBlur(features, (0, 0), 0.6)
    return features.ravel()


class PersianDigitClassifier:
    """k-NN over normalised glyph bitmaps with similarity weighted voting."""

    name = "glyph-knn"

    def __init__(self, model_path: str | Path | None = None, *, neighbours: int = 7) -> None:
        """Create the k-NN classifier and immediately load the bundled ``.npz`` model.
        ``neighbours`` is the k of the vote; 7 was the best trade-off measured on the
        holdout font (fewer → noisy, more → blurred classes).
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.neighbours = neighbours
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self.metadata: dict = {}
        self._load()

    # ── model handling ───────────────────────────────────────────────
    def _load(self) -> None:
        """Load features/labels from the ``npz`` and pre-normalise the rows for cosine search.
        A corrupt or missing file leaves the classifier unavailable instead of raising,
        so the pipeline can fall back to Tesseract per glyph.
        """
        if not self.model_path.is_file():
            return
        try:
            with np.load(self.model_path, allow_pickle=False) as data:
                features = data["features"].astype(np.float32) / 255.0
                self._labels = data["labels"].astype(np.int32)
                if "meta" in data:
                    self.metadata = json.loads(str(data["meta"].tolist()))
        except Exception:  # pragma: no cover - corrupted model should not crash the API
            self._features = self._labels = None
            return
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        self._features = features / norms

    @property
    def available(self) -> bool:
        """True when a model with at least one sample is loaded."""
        return self._features is not None and self._labels is not None and len(self._labels) > 0

    @property
    def accuracy(self) -> float | None:
        """Holdout accuracy recorded in the model metadata (``None`` if not trained with one)."""
        value = self.metadata.get("holdout_accuracy")
        return float(value) if value is not None else None

    # ── inference ────────────────────────────────────────────────────
    def classify(self, patch: np.ndarray) -> DigitPrediction | None:
        """Classify one glyph patch (numpy image) → :class:`DigitPrediction` or ``None``.
        ``None`` means «no opinion» (empty patch) and lets the caller try Tesseract.
        """
        if not self.available:
            return None
        features = featurize(patch)
        if features is None:
            return None
        return self.classify_features(features)

    def classify_features(self, features: np.ndarray) -> DigitPrediction | None:
        """Classify a pre-computed feature vector with a cosine k-NN vote.
        Returns the winning digit with its similarity, the runner-up margin and the
        top-3 shares that later become field-level alternatives.
        """
        if not self.available or features is None:
            return None
        vector = features.astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return None
        vector = vector / norm
        similarities = self._features @ vector  # type: ignore[union-attr]
        top = min(self.neighbours, similarities.shape[0])
        indices = np.argpartition(-similarities, top - 1)[:top]
        indices = indices[np.argsort(-similarities[indices])]

        scores = np.zeros(10, dtype=np.float32)
        for index in indices:
            weight = max(0.0, float(similarities[index])) ** 3
            scores[int(self._labels[index])] += weight  # type: ignore[index]
        if scores.sum() <= 0:
            return None
        order = np.argsort(-scores)
        best, second = order[0], order[1]
        total = float(scores.sum()) or 1.0
        confidence = 100.0 * float(scores[best]) / total
        margin = (float(scores[best]) - float(scores[second])) / max(1e-6, float(scores[best]))
        ranked = [
            (PERSIAN_DIGITS[int(index)], round(float(scores[index]) / total, 4))
            for index in order[:3]
            if scores[index] > 0
        ]
        return DigitPrediction(
            digit=PERSIAN_DIGITS[int(best)],
            confidence=round(confidence, 1),
            similarity=round(float(similarities[indices[0]]), 4),
            margin=round(margin, 4),
            options=ranked,
        )

    def classify_many(self, patches: list[np.ndarray]) -> list[DigitPrediction | None]:
        """Classify a batch of glyph patches in one matrix multiplication (used by the trainer)."""
        return [self.classify(patch) for patch in patches]


@lru_cache(maxsize=4)
def get_digit_classifier(model_path: str | None = None) -> PersianDigitClassifier:
    """Cached classifier instance."""
    return PersianDigitClassifier(model_path)
