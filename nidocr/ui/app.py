"""NIDOCR — a self-contained, RTL Streamlit UI for the Iranian national card.

Run it with::

    streamlit run nidocr/ui/app.py
    # or
    nidocr ui

Design notes
------------
* **Self-contained.** The streamlit iframe is often rendered in an environment
  without outbound network access, so nothing is loaded from a CDN: the Vazirmatn
  font is embedded as base64, the icons are inline SVG/emoji and the pipeline runs
  in-process (no HTTP call to the API).
* **Persian first.** Everything is right-to-left, values are printed with Persian
  digits, and the layout uses the same vocabulary as the card itself.
* **Honest.** Every value shows its confidence, whether the validator accepted it,
  and the raw OCR string it came from.
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # allow `streamlit run nidocr/ui/app.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from nidocr import __version__  # noqa: E402
from nidocr.accuracy import (  # noqa: E402
    AccuracyReport,
    compare_card,
    estimate_card,
    load_dataset,
    run_synthetic,
)
from nidocr.fields.definitions import FIELD_SPECS  # noqa: E402
from nidocr.fields.lexicon import lexicon_status  # noqa: E402
from nidocr.image_io import load_image  # noqa: E402
from nidocr.ocr.text import to_persian_digits  # noqa: E402
from nidocr.pipeline import get_ocr  # noqa: E402
from nidocr.vision.layout import available_templates, load_template  # noqa: E402

FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"
PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

PALETTE = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "line": "rgba(148, 163, 184, 0.28)",
    "good": "#12805c",
    "warn": "#b45309",
    "bad": "#b91c1c",
    "accent": "#2563eb",
}

st.set_page_config(
    page_title="NIDOCR · کارت ملی",
    page_icon="🪪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── assets ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def font_css() -> str:
    """Embed Vazirmatn as base64 so the UI never depends on a CDN."""
    face = []
    for name, weight in (("Vazirmatn-Regular.ttf", 400), ("Vazirmatn-Medium.ttf", 500), ("Vazirmatn-Bold.ttf", 700)):
        path = FONTS_DIR / name
        if not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode()
        face.append(
            f"@font-face{{font-family:'Vazirmatn';font-style:normal;font-weight:{weight};"
            f"font-display:swap;src:url(data:font/ttf;base64,{encoded}) format('truetype');}}"
        )
    return "\n".join(face)


def style_block() -> str:
    """The full CSS of the interface, with Vazirmatn embedded as base64 (no CDN needed)."""
    return f"""
<style>
{font_css()}

html, body, [class*="css"], .stApp {{
    font-family: 'Vazirmatn', 'Segoe UI', sans-serif;
}}
.stApp {{
    background:
        radial-gradient(1100px 520px at 88% -12%, rgba(37, 99, 235, 0.16), transparent 60%),
        radial-gradient(900px 460px at 6% 0%, rgba(16, 185, 129, 0.13), transparent 55%),
        linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0b1220 0%, #101a2c 100%);
    border-left: 1px solid rgba(255,255,255,0.06);
}}
[data-testid="stSidebar"] * {{ color: #e2e8f0; }}
[data-testid="stSidebar"] .stButton button {{
    background: rgba(255,255,255,0.08); color:#e2e8f0; border:1px solid rgba(255,255,255,0.14);
}}
[data-testid="stSidebar"] .stButton button:hover {{ background: rgba(255,255,255,0.18); }}

.block-container {{ padding-top: 1.6rem; max-width: 1360px; direction: rtl; }}

.hero {{
    position: relative; overflow: hidden; border-radius: 26px; padding: 28px 32px;
    background: linear-gradient(120deg, #0b1220 0%, #16233c 45%, #1e3a8a 100%);
    color: #f8fafc; box-shadow: 0 22px 48px -24px rgba(15, 23, 42, 0.75);
    margin-bottom: 22px;
}}
.hero::after {{
    content: ""; position: absolute; inset: auto -60px -120px auto; width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.35), transparent 65%);
}}
.hero h1 {{ margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -.4px; }}
.hero p {{ margin: 8px 0 0; color: #c7d2fe; font-size: .95rem; }}
.hero .chips {{ margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{
    font-size: .78rem; padding: 5px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.18);
}}

.card-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(232px, 1fr)); gap: 14px; }}
.field-card {{
    background: #ffffff; border-radius: 18px; padding: 16px 18px;
    border: 1px solid {PALETTE["line"]}; box-shadow: 0 14px 30px -26px rgba(15,23,42,.85);
    transition: transform .15s ease, box-shadow .15s ease;
}}
.field-card:hover {{ transform: translateY(-2px); box-shadow: 0 20px 36px -26px rgba(15,23,42,.9); }}
.field-label {{ font-size: .8rem; color: {PALETTE["muted"]}; display:flex; justify-content:space-between; gap:6px; }}
.field-value {{ font-size: 1.45rem; font-weight: 700; color: {PALETTE["ink"]}; margin: 6px 0 2px; word-break: break-word; }}
.field-value.empty {{ color: #cbd5e1; font-weight: 500; }}
.field-value.mono {{ font-family: 'Vazirmatn', ui-monospace, monospace; letter-spacing: .5px; }}
.field-raw {{ font-size: .74rem; color: #94a3b8; min-height: 1rem; }}
.bar {{ height: 6px; border-radius: 999px; background: #e2e8f0; overflow: hidden; margin-top: 10px; }}
.bar > span {{ display: block; height: 100%; border-radius: 999px; }}
.badge {{
    font-size: .68rem; padding: 3px 9px; border-radius: 999px; font-weight: 600; white-space: nowrap;
}}
.badge.ok {{ background: #dcfce7; color: {PALETTE["good"]}; }}
.badge.warn {{ background: #fef3c7; color: {PALETTE["warn"]}; }}
.badge.bad {{ background: #fee2e2; color: {PALETTE["bad"]}; }}
.badge.neutral {{ background: #e2e8f0; color: #475569; }}

.kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0 18px; }}
.kpi > div {{
    background: rgba(255,255,255,.85); border: 1px solid {PALETTE["line"]}; border-radius: 16px;
    padding: 14px 16px; backdrop-filter: blur(6px);
}}
.kpi b {{ display: block; font-size: 1.5rem; color: {PALETTE["ink"]}; }}
.kpi span {{ font-size: .76rem; color: {PALETTE["muted"]}; }}

.notes {{ background: rgba(255,255,255,.8); border: 1px solid {PALETTE["line"]}; border-radius: 16px; padding: 14px 18px; }}
.notes li {{ font-size: .85rem; color: #475569; margin: 4px 0; }}

.acc-wrap {{ background: #ffffff; border: 1px solid {PALETTE["line"]}; border-radius: 18px; overflow: hidden; }}
table.acc {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
table.acc th, table.acc td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #eef2f7; }}
table.acc th {{ background: #f8fafc; color: {PALETTE["muted"]}; font-weight: 600; font-size: .78rem; }}
table.acc tr:last-child td {{ border-bottom: none; }}
table.acc td.num {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
table.acc td .track {{ height: 7px; border-radius: 999px; background: #e2e8f0; overflow: hidden; min-width: 90px; }}
table.acc td .track > span {{ display: block; height: 100%; border-radius: 999px; }}
.pill {{ display: inline-block; font-size: .72rem; padding: 2px 9px; border-radius: 999px; background: #e2e8f0; color: #475569; }}
.pill.est {{ background: #dbeafe; color: #1d4ed8; }}
.pill.meas {{ background: #dcfce7; color: #12805c; }}
.pill.miss {{ background: #fee2e2; color: #b91c1c; }}

[data-testid="stFileUploaderDropzone"] {{
    background: rgba(255,255,255,.72); border: 1.5px dashed #93c5fd; border-radius: 20px;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 12px 12px 0 0; padding: 8px 16px; background: rgba(255,255,255,.6);
}}
.stTabs [aria-selected="true"] {{ background: #ffffff; font-weight: 600; }}
div[data-testid="stMetricValue"] {{ font-size: 1.4rem; }}
.stDownloadButton button, .stButton button {{ border-radius: 12px; }}
footer, #MainMenu {{ visibility: hidden; }}
</style>
"""


# ── helpers ──────────────────────────────────────────────────────────────────
def badge(field) -> tuple[str, str]:
    """Map a field result to a badge ``(css_class, persian_text)``: valid / repaired / review / missing."""
    if not field.found:
        return "bad", "یافت نشد"
    if field.repaired:
        return "warn", "اصلاح‌شده"
    if field.validated:
        return "ok", "معتبر"
    return "neutral", "بازبینی"


def field_card_html(field) -> str:
    """Render one field as a gradient card: label, value in Persian digits, raw OCR, confidence bar."""
    tone, text = badge(field)
    value = field.value_fa or (to_persian_digits(field.value) if field.value else "")
    numeric = field.kind in {"digits", "date", "serial"}
    confidence = max(0.0, min(100.0, field.confidence))
    colour = PALETTE["good"] if field.validated else PALETTE["warn"] if field.found else PALETTE["bad"]
    raw = field.raw_value if field.raw_value and field.raw_value != field.value else ""
    return f"""
<div class="field-card">
  <div class="field-label"><span>{field.label_fa}</span><span class="badge {tone}">{text}</span></div>
  <div class="field-value {'mono' if numeric else ''} {'empty' if not value else ''}">{value or '—'}</div>
  <div class="field-raw">{('خام: ' + raw) if raw else '&nbsp;'}</div>
  <div class="bar"><span style="width:{confidence:.0f}%;background:{colour}"></span></div>
  <div class="field-label" style="margin-top:6px"><span>اطمینان</span><span>{confidence:.0f}٪</span></div>
</div>
"""


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a numpy image as PNG bytes for ``st.image``."""
    import cv2

    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes() if ok else b""


def b64_image(payload: str | None) -> bytes | None:
    """Turn a base64 ROI/overlay string from the API model into bytes."""
    if not payload:
        return None
    try:
        return base64.b64decode(payload.split(",")[-1])
    except Exception:  # pragma: no cover - defensive
        return None


@st.cache_resource(show_spinner="در حال آماده‌سازی موتور OCR…")
def pipeline(template: str, upscale: float, passes: int, anchors: bool):
    """One pipeline per (template, tuning) combination."""
    from nidocr.config import Settings

    settings = Settings(upscale=upscale, passes=passes, use_label_anchors=anchors, default_template=template)
    return get_ocr(settings)


@st.cache_data(show_spinner=False)
def demo_card(seed: int) -> bytes:
    """Render a synthetic card so the UI can be tried without any upload."""
    from nidocr.synthetic.renderer import random_spec, render_card

    layout = load_template("national_card_front")
    image = render_card(random_spec(seed), layout)
    ok, buffer = __import__("cv2").imencode(".jpg", image, [int(__import__("cv2").IMWRITE_JPEG_QUALITY), 92])
    return buffer.tobytes() if ok else b""


def quality_notes(result) -> list[str]:
    """Persian advice derived from the quality report (blur, exposure, glare, resolution)."""
    notes = list(result.warnings)
    quality = result.quality
    if quality.sharpness < 60:
        notes.append("تصویر مات است؛ عکس تازه و بدون لرزش بگیرید.")
    if quality.brightness < 70:
        notes.append("نور کم است؛ کارت را در نور یکنواخت قرار دهید.")
    if quality.brightness > 210:
        notes.append("تصویر بیش از حد روشن است؛ بازتاب نور را حذف کنید.")
    if quality.glare_ratio > 0.02:
        notes.append("بازتاب نور روی کارت دیده می‌شود.")
    if quality.rotation_degrees:
        notes.append(f"تصویر {to_persian_digits(str(quality.rotation_degrees))} درجه چرخانده شد تا متن افقی شود.")
    if not quality.card_detected:
        notes.append("کادر کارت با اطمینان پیدا نشد؛ کل تصویر پردازش شد.")
    if quality.resolution[0] < 700:
        notes.append("تفکیک‌پذیری تصویر پایین است.")
    return notes


# ── accuracy ─────────────────────────────────────────────────────────────────
TEMPLATE_LABELS = {
    "national_card_front": "کارت ملی (قدیمی) — front",
    "national_card_back": "کارت ملی (قدیمی) — back",
    "national_card_smart": "کارت هوشمند ملی — front",
}

DIFFICULTY_LABELS = {
    "clean": "اسکن تخت (clean)",
    "photo": "عکس با دوربین (photo)",
    "hard": "عکس دشوار (hard)",
}


def _acc_colour(rate: float) -> str:
    """Green above 90 %, amber above 70 %, red below — for a rate in ``0..1``."""
    if rate >= 0.90:
        return PALETTE["good"]
    if rate >= 0.70:
        return PALETTE["warn"]
    return PALETTE["bad"]


def accuracy_table_html(rows: list[dict], *, estimated: bool) -> str:
    """Build the per-field accuracy table (label, exact %, char %, confidence, bar).

    ``rows`` entries need ``label_fa``, ``exact`` and ``characters`` as rates in
    ``0..1`` plus ``confidence``; ``note`` is optional extra text (for estimates:
    *why* the number was adjusted). The header names the metric honestly — the
    caller decides whether the numbers are measured or estimated.
    """
    head = (
        "<tr><th>فیلد</th><th>تطابق کامل</th><th>دقت کاراکتر</th>"
        "<th>اطمینان</th><th>نمودار</th><th>یادداشت</th></tr>"
    )
    body = []
    for row in rows:
        exact = float(row.get("exact", 0.0))
        chars = float(row.get("characters", 0.0))
        body.append(
            "<tr>"
            f"<td>{row.get('label_fa', '')}</td>"
            f"<td class='num' style='color:{_acc_colour(exact)}'>{exact * 100:.1f}٪</td>"
            f"<td class='num'>{chars * 100:.1f}٪</td>"
            f"<td class='num'>{float(row.get('confidence', 0.0)):.0f}٪</td>"
            f"<td><div class='track'><span style='width:{exact * 100:.0f}%;"
            f"background:{_acc_colour(exact)}'></span></div></td>"
            f"<td style='color:{PALETTE['muted']};font-size:.76rem'>{row.get('note', '')}</td>"
            "</tr>"
        )
    tag = "تخمینی" if estimated else "اندازه‌گیری‌شده"
    pill = "est" if estimated else "meas"
    return (
        f"<div class='acc-wrap'><table class='acc'>{head}{''.join(body)}</table></div>"
        f"<div style='margin-top:6px'><span class='pill {pill}'>{tag}</span></div>"
    )


def accuracy_kpi_html(report: AccuracyReport) -> str:
    """KPI strip of a measured report: exact, characters, perfect cards, latency."""
    return f"""
<div class="kpi">
  <div><b>{report.exact * 100:.1f}٪</b><span>تطابق کامل فیلدها</span></div>
  <div><b>{report.characters * 100:.1f}٪</b><span>دقت کاراکتر</span></div>
  <div><b>{report.perfect_cards * 100:.0f}٪</b><span>کارت‌های بی‌غلط</span></div>
  <div><b>{report.mean_latency_ms:.0f} ms</b><span>زمان هر کارت</span></div>
  <div><b>{report.cards}</b><span>کارت آزمون</span></div>
</div>
"""


def render_measured_report(report: AccuracyReport, *, key_prefix: str) -> None:
    """Render a measured :class:`AccuracyReport`: KPIs, per-field table, chart, downloads."""
    st.markdown(accuracy_kpi_html(report), unsafe_allow_html=True)
    rows = [
        {
            "label_fa": row.label_fa,
            "exact": row.exact,
            "characters": row.characters,
            "confidence": row.confidence,
            "note": f"اعتبارسنجی {row.validated * 100:.0f}٪ · {row.samples} نمونه",
        }
        for row in report.fields
    ]
    st.markdown(accuracy_table_html(rows, estimated=False), unsafe_allow_html=True)
    st.markdown("#### نمودار تطابق کامل هر فیلد")
    st.bar_chart({row.label_fa: round(row.exact * 100, 1) for row in report.fields}, horizontal=True, height=250)
    if report.failures:
        with st.expander(f"خطاها ({len(report.failures)})"):
            st.dataframe(
                [item.as_row() for item in report.failures],
                width="stretch",
                hide_index=True,
            )
    cols = st.columns(2)
    cols[0].download_button(
        "دانلود گزارش (Markdown)",
        data=report.to_markdown(title="NIDOCR accuracy").encode(),
        file_name=f"nidocr-accuracy-{report.difficulty}.md",
        mime="text/markdown",
        key=f"{key_prefix}-acc-md",
    )
    cols[1].download_button(
        "دانلود متریک‌ها (JSON)",
        data=json.dumps(report.as_dict(), ensure_ascii=False, indent=2).encode(),
        file_name=f"nidocr-accuracy-{report.difficulty}.json",
        mime="application/json",
        key=f"{key_prefix}-acc-json",
    )
    st.caption(
        f"منبع: {report.cards} کارت با حقیقت معلوم · سطح دشواری: {DIFFICULTY_LABELS.get(report.difficulty, report.difficulty)}"
    )


def accuracy_panel(engine) -> None:
    """The «سنجش دقت» expander: measure the model on synthetic cards or your own dataset.

    Two sources are offered because they answer different questions: *synthetic*
    cards give a reproducible number for the model itself, while *my dataset*
    measures the actual scanned cards (with their real fonts, printing and light).
    """
    st.markdown(
        "دقت **هر فیلد** و **کل کارت** را اینجا اندازه بگیرید. برای کارت‌های ساختگی، "
        "حقیقت از پیش معلوم است؛ برای پوشهٔ خودتان به ازای هر تصویر یک فایل JSON با کلید "
        "`fields` لازم است."
    )
    controls = st.columns([1.1, 1, 0.8, 0.8])
    source = controls[0].radio(
        "منبع آزمون",
        ["کارت‌های ساختگی", "پوشهٔ داده‌های من"],
        horizontal=False,
        key="acc-source",
    )
    difficulty = controls[1].selectbox(
        "سطح دشواری",
        list(DIFFICULTY_LABELS),
        index=0,
        format_func=lambda key: DIFFICULTY_LABELS[key],
        key="acc-difficulty",
    )
    count = controls[2].number_input("تعداد کارت", 2, 40, 8, step=2, key="acc-count")
    seed = controls[3].number_input("بذر", 1, 9999, 4242, step=1, key="acc-seed")

    folder = ""
    if source == "پوشهٔ داده‌های من":
        folder = st.text_input(
            "مسیر پوشه (تصویر + JSON)",
            value="/tmp/final_clean",
            key="acc-folder",
            help="هر تصویر باید یک JSON هم‌نام با کلید «fields» داشته باشد.",
        )

    if st.button("📊 اندازه‌گیری دقت", width="stretch", key="acc-run"):
        bar = st.progress(0.0, text="در حال آماده‌سازی…")

        def advance(done: int, total: int, label: str) -> None:
            """Update the progress bar after every card."""
            bar.progress(min(done / max(total, 1), 1.0), text=f"{done}/{total} · {label}")

        try:
            if source == "کارت‌های ساختگی":
                report = run_synthetic(
                    count=int(count),
                    difficulty=difficulty,
                    seed=int(seed),
                    ocr=engine,
                    progress=advance,
                )
            else:
                records = load_dataset(folder)
                if not records:
                    st.error("در این پوشه تصویر + JSON پیدا نشد.")
                    st.stop()
                from nidocr.accuracy import evaluate

                report = evaluate(records[: int(count)], ocr=engine, difficulty=difficulty, progress=advance)
            bar.progress(1.0, text="پایان")
            st.session_state["accuracy_report"] = report
        except Exception as error:  # pragma: no cover - surfaced to the user
            st.error(f"اندازه‌گیری ناموفق بود: {error}")
            st.session_state.pop("accuracy_report", None)

    report = st.session_state.get("accuracy_report")
    if isinstance(report, AccuracyReport):
        render_measured_report(report, key_prefix="panel")


def render_estimate(result, difficulty: str, key_prefix: str) -> None:
    """Per-field **and** overall accuracy of the current card, estimated without ground truth.

    Nothing here is measured on this image: the prior is the committed benchmark
    rate of each field, adjusted by the field's own confidence/validator/repair
    state. The «تخمینی» pill makes that explicit, and the ground-truth form below
    replaces it with a real measurement when the user has the true values.
    """
    estimate = estimate_card(result.ordered_fields(), difficulty=difficulty)
    st.markdown(
        f"""
<div class="kpi">
  <div><b>{estimate['exact'] * 100:.1f}٪</b><span>دقت کل کارت (تخمینی)</span></div>
  <div><b>{estimate['characters'] * 100:.1f}٪</b><span>دقت کاراکتر (تخمینی)</span></div>
  <div><b>{DIFFICULTY_LABELS.get(difficulty, difficulty)}</b><span>شرایط فرض‌شده</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    rows = [row.as_row() for row in estimate["fields"]]
    st.markdown(accuracy_table_html(rows, estimated=True), unsafe_allow_html=True)
    st.bar_chart({row["label_fa"]: row["exact"] for row in rows}, horizontal=True, height=240)
    st.caption(
        "این اعداد از نرخ اندازه‌گیری‌شدهٔ هر فیلد در گزارش‌های `reports/` شروع می‌شوند و با "
        "اطمینان و نتیجهٔ اعتبارسنجی همین کارت تنظیم می‌شوند؛ پس تخمینی‌اند، نه اندازه‌گیری."
    )


def render_truth_form(result, key_prefix: str) -> None:
    """Ask for the true values of this card and measure exact/character accuracy per field.

    The inputs are pre-filled with what the engine read, so the user only corrects
    the wrong fields; the comparison is then made against the corrected values.
    """
    st.markdown("#### اندازه‌گیری واقعی این کارت")
    st.caption("مقدارهای درست را اصلاح کنید و دکمه را بزنید؛ خالی گذاشتن یک فیلد آن را نادیده می‌گیرد.")
    keys = [field.key for field in result.ordered_fields()]
    with st.form(f"{key_prefix}-truth-form"):
        columns = st.columns(2)
        truth: dict[str, str] = {}
        for index, key in enumerate(keys):
            field = result.fields.get(key)
            truth[key] = columns[index % 2].text_input(
                FIELD_SPECS[key].label_fa if key in FIELD_SPECS else key,
                value=(field.value if field else "") or "",
                key=f"{key_prefix}-truth-{key}",
            )
        submitted = st.form_submit_button("محاسبهٔ دقت این کارت", width="stretch")

    if not submitted:
        return
    score = compare_card(result, truth)
    if not score["measured"]:
        st.warning("هیچ مقداری وارد نشد.")
        return
    st.markdown(
        f"""
<div class="kpi">
  <div><b>{score['correct']}/{score['measured']}</b><span>فیلد کاملاً درست</span></div>
  <div><b>{score['exact_rate'] * 100:.1f}٪</b><span>تطابق کامل</span></div>
  <div><b>{score['characters'] * 100:.1f}٪</b><span>دقت کاراکتر</span></div>
</div>
""",
        unsafe_allow_html=True,
    )
    rows = [
        {
            "label_fa": row["label_fa"],
            "exact": 1.0 if row["exact"] else 0.0,
            "characters": row["characters"],
            "confidence": row["confidence"],
            "note": "✔ درست" if row["exact"] else f"✘ خوانده‌شده: {row['predicted'] or '—'}",
        }
        for row in score["fields"]
    ]
    st.markdown(accuracy_table_html(rows, estimated=False), unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────
def sidebar():
    """Draw the sidebar controls and return ``(template, upscale, passes, anchors, debug, condition)``."""
    with st.sidebar:
        st.markdown(
            "<div style='padding:4px 2px 14px'>"
            "<div style='font-size:1.25rem;font-weight:700'>🪪 NIDOCR</div>"
            f"<div style='color:#94a3b8;font-size:.78rem'>نسخهٔ {__version__} · OCR کارت ملی</div></div>",
            unsafe_allow_html=True,
        )

        templates = ["auto", *available_templates()]
        template = st.selectbox(
            "قالب کارت",
            templates,
            index=0,
            format_func=lambda name: "تشخیص خودکار" if name == "auto" else TEMPLATE_LABELS.get(name, name),
        )
        st.caption(
            "«تشخیص خودکار» طرح کارت را از متن چاپ‌شده می‌شناسد "
            "(کارت قدیمی با سریال/جنسیت یا کارت هوشمند با پایان اعتبار)."
        )

        condition = st.selectbox(
            "شرایط تصویر (برای تخمین دقت)",
            ["photo", "clean", "hard"],
            index=0,
            format_func=lambda key: DIFFICULTY_LABELS[key],
            help="مبنای تخمین دقت هر فیلد؛ روی خواندن اثر ندارد.",
        )

        with st.expander("تنظیمات پیشرفته", expanded=False):
            upscale = st.slider("بزرگ‌نمایی تصویر", 2.0, 4.0, 3.0, 0.5)
            passes = st.slider("تعداد پاس OCR", 1, 4, 3)
            anchors = st.toggle("استفاده از جای‌نگهدار برچسب‌ها", value=True)
            debug = st.toggle("تصاویر عیب‌یابی (کادرها و ناحیه‌ها)", value=True)

        st.divider()
        engine_state = "…"
        try:
            from nidocr.ocr.tesseract_engine import get_engine

            engine = get_engine()
            ready, message = engine.ready()
            engine_state = f"{'✅ آماده' if ready else '⚠️ ناقص'} · {engine.name}"
            if not ready:
                st.warning(message)
        except Exception as exc:  # pragma: no cover - defensive
            engine_state = f"⚠️ {exc}"
        lexicon = lexicon_status()
        st.markdown(
            f"<div style='font-size:.8rem;color:#cbd5e1'>موتور: {engine_state}</div>"
            f"<div style='font-size:.8rem;color:#cbd5e1'>فرهنگ نام‌ها: "
            f"{lexicon.get('first_names', 0):,} نام · {lexicon.get('last_names', 0):,} نام خانوادگی</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<div style='font-size:.75rem;color:#94a3b8;line-height:1.7'>"
            "داده‌های شما جایی ارسال نمی‌شود؛ همهٔ پردازش در همین سرور انجام می‌شود."
            "</div>",
            unsafe_allow_html=True,
        )
    return template, upscale, passes, anchors, debug, condition


# ── result rendering ─────────────────────────────────────────────────────────
def render_result(
    result,
    source_image: np.ndarray | None,
    debug: bool,
    key_prefix: str,
    difficulty: str = "photo",
) -> None:
    """Render a whole card: KPIs, annotated image, field cards and the JSON/quality/ROI/accuracy tabs.

    ``difficulty`` is the image condition the accuracy estimate should assume
    (chosen in the sidebar); the measured panel lives above the uploader.
    """
    fields = [result.fields[k] for k in (result.field_order or result.fields) if k in result.fields]
    found = [f for f in fields if f.found]
    validated = [f for f in found if f.validated]
    mean_conf = result.mean_confidence
    estimate = estimate_card(result.ordered_fields(), difficulty=difficulty)

    st.markdown(
        f"""
<div class="kpi">
  <div><b>{len(validated)}/{len(fields)}</b><span>فیلد تأییدشده</span></div>
  <div><b>{estimate['exact'] * 100:.1f}٪</b><span>دقت کل (تخمینی)</span></div>
  <div><b>{mean_conf:.0f}٪</b><span>میانگین اطمینان</span></div>
  <div><b>{result.processing_ms:.0f} ms</b><span>زمان پردازش</span></div>
  <div><b>{result.quality.sharpness:.0f}</b><span>وضوح تصویر</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("#### 🖼️ تصویر کارت")
        payload = b64_image(result.overlay_png) if debug else None
        if payload is None and source_image is not None:
            payload = image_to_png_bytes(source_image)
        if payload:
            st.image(payload, width="stretch")
        if debug and result.processed_png:
            with st.expander("تصویر پردازش‌شده (پیش‌پردازش)"):
                processed = b64_image(result.processed_png)
                if processed:
                    st.image(processed, width="stretch")
    with right:
        st.markdown("#### 🧾 متن استخراج‌شده")
        for field in fields:
            value = field.value_fa or to_persian_digits(field.value)
            row = st.container()
            cols = row.columns([1.1, 1.6, 0.7])
            cols[0].caption(field.label_fa)
            cols[1].markdown(f"**{value or '—'}**")
            tone, label = badge(field)
            cols[2].caption(f"{field.confidence:.0f}٪")
            if field.message:
                row.caption(f"↳ {field.message}")

    st.markdown("#### 🗂️ کارت‌های فیلد")
    st.markdown(
        "<div class='card-grid'>" + "".join(field_card_html(f) for f in fields) + "</div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📄 JSON", "🧠 تحلیل اطمینان", "🖼️ ناحیه‌های فیلد", "🎯 دقت این کارت"])
    with tabs[0]:
        st.code(json.dumps(result.model_dump(exclude={"overlay_png", "processed_png"}), ensure_ascii=False, indent=2), language="json")
        st.download_button(
            "دانلود JSON",
            data=json.dumps(result.model_dump(exclude={"overlay_png", "processed_png"}), ensure_ascii=False, indent=2).encode(),
            file_name=f"nidocr-{int(time.time())}.json",
            mime="application/json",
            key=f"{key_prefix}-dl",
        )
    with tabs[1]:
        chart = {f.label_fa: f.confidence for f in fields}
        st.bar_chart(chart, horizontal=True, height=260)
        notes = quality_notes(result)
        if notes:
            st.markdown("#### نکته‌ها")
            st.markdown("<div class='notes'><ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul></div>", unsafe_allow_html=True)
        st.caption(f"قالب: {result.template_title_fa} · موتور: {result.engine} ({result.languages})")
    with tabs[2]:
        rows = [f for f in fields if f.roi_png]
        if not rows:
            st.info("برای دیدن ناحیه‌ها، گزینهٔ «تصاویر عیب‌یابی» را روشن کنید.")
        for chunk_start in range(0, len(rows), 4):
            cols = st.columns(4, gap="small")
            for column, field in zip(cols, rows[chunk_start : chunk_start + 4], strict=False):
                with column:
                    st.caption(f"{field.label_fa} · {field.confidence:.0f}٪")
                    st.image(b64_image(field.roi_png), width="stretch")
    with tabs[3]:
        st.markdown("#### دقت این کارت — به تفکیک فیلد و در کل")
        render_estimate(result, difficulty, key_prefix)
        st.divider()
        render_truth_form(result, key_prefix)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    """Page entry point: style, sidebar, upload/demo input, then extract and render the card."""
    st.markdown(style_block(), unsafe_allow_html=True)
    template, upscale, passes, anchors, debug, condition = sidebar()

    st.markdown(
        """
<div class="hero">
  <h1>خواندن هوشمند کارت ملی ایران</h1>
  <p>تشخیص کارت با OpenCV، خواندن فیلدها با موتور فارسی و اعتبارسنجی با چک‌سام کد ملی و تقویم جلالی.</p>
  <div class="chips">
    <span class="chip">تشخیص خودکار قاب کارت</span>
    <span class="chip">فیلدهای نام، نام خانوادگی، نام پدر</span>
    <span class="chip">کد ملی · تاریخ تولد · شماره سریال · جنسیت</span>
    <span class="chip">API مستقل برای هر فیلد</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    engine = pipeline(template, float(upscale), int(passes), bool(anchors))

    with st.expander("📊 سنجش دقت مدل روی چند کارت (اندازه‌گیری واقعی)", expanded=False):
        accuracy_panel(engine)

    upload_col, sample_col = st.columns([2.2, 1], gap="large")
    with upload_col:
        uploaded = st.file_uploader(
            "تصویر کارت ملی را اینجا رها کنید",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=False,
        )
        camera = st.camera_input("یا با دوربین عکس بگیرید", label_visibility="collapsed")
    with sample_col:
        st.markdown("#### بدون تصویر؟")
        st.caption("یک کارت نمونهٔ ساختگی بسازید و مسیر کامل را آزمایش کنید.")
        seed = st.number_input("بذر نمونه", 1, 9999, 7, step=1)
        make_demo = st.button("ساخت کارت نمونه", width="stretch")

    if make_demo:
        # keep the demo card across reruns, otherwise every widget interaction
        # (sidebar change, accuracy form submit, …) would clear the result.
        st.session_state["demo_payload"] = demo_card(int(seed))
        st.session_state["demo_name"] = f"sample-{int(seed)}"

    payload = None
    filename = "card"
    if uploaded is not None:
        payload = uploaded.getvalue()
        filename = Path(uploaded.name).stem
    elif camera is not None:
        payload = camera.getvalue()
        filename = "camera"
    elif st.session_state.get("demo_payload"):
        payload = st.session_state["demo_payload"]
        filename = st.session_state.get("demo_name", "sample")

    if payload is None:
        st.info("برای شروع یک تصویر بارگذاری کنید یا روی «ساخت کارت نمونه» بزنید.")
        st.stop()

    image = load_image(payload)
    st.image(image, caption=f"{filename} · {image.shape[1]}×{image.shape[0]}", width="stretch")

    with st.spinner("در حال خواندن کارت…"):
        started = time.perf_counter()
        result = engine.extract(image, template=template, debug=bool(debug))
        elapsed = (time.perf_counter() - started) * 1000

    if not result.success:
        st.error("هیچ فیلدی خوانده نشد. تصویر را با نور بهتر و بدون بازتاب امتحان کنید.")
    results = st.session_state.setdefault("history", [])
    results.insert(0, {"name": filename, "ms": round(elapsed), "result": result})

    tab_current, tab_history, tab_about = st.tabs(["نتیجهٔ فعلی", "تاریخچهٔ این نشست", "درباره"])
    with tab_current:
        render_result(result, image, bool(debug), key_prefix="current", difficulty=condition)
    with tab_history:
        if not results:
            st.caption("هنوز کارتی پردازش نشده است.")
        for index, item in enumerate(results[:8]):
            with st.expander(f"{index + 1}. {item['name']} · {item['ms']} ms"):
                st.markdown(
                    "<div class='card-grid'>" + "".join(field_card_html(f) for f in item["result"].ordered_fields()) + "</div>",
                    unsafe_allow_html=True,
                )
    with tab_about:
        st.markdown(
            f"""
**NIDOCR {__version__}** — خواندن کارت ملی هوشمند ایران با Python و OpenCV.

* پیش‌پردازش: حذف گرادیان و گیلوش، CLAHE، شارپ کردن، آستانه‌گذاری Sauvola/Otsu
* تشخیص: پیدا کردن قاب کارت و تبدیل پرسپکتیو به بوم استاندارد ۱۰۱۲×۶۳۸
* خواندن اعداد: قطعه‌بندی گلیف‌ها + طبقه‌بند k-NN فارسی + رأی‌گیری دو باینری‌سازی
* خواندن نام‌ها: موتور Tesseract فارسی (psm 7 + حالت‌های پشتیبان) + فرهنگ نام‌ها
* اعتبارسنجی: چک‌سام mod-11 کد ملی، تقویم جلالی، واژه‌نامهٔ نام
* چرخش: تصحیح خودکار ۹۰/۱۸۰/۲۷۰ درجه با OSD و بازخوانی نجات‌بخش
* دقت: اندازه‌گیری واقعی هر فیلد و کل کارت (`nidocr/accuracy.py`) + تخمین برای کارت‌های بدون حقیقت

فیلدهای قابل خواندن: {", ".join(spec.label_fa for spec in FIELD_SPECS.values())}
"""
        )


if __name__ == "__main__":
    main()
