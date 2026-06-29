import os
from core.config import settings

# Must set these environment variables before importing paddle/paddleocr!
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = os.getenv("FLAGS_fraction_of_gpu_memory_to_use", "0.1")

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy imports — torch and paddleocr may not be installed on lightweight deployments
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    logger.warning("torch not installed — GPU detection disabled, using CPU fallback.")

try:
    from paddleocr import PaddleOCR
    _HAS_PADDLEOCR = True
except ImportError:
    PaddleOCR = None  # type: ignore
    _HAS_PADDLEOCR = False
    logger.warning("paddleocr not installed — OCR functionality will be unavailable.")


def _detect_gpu() -> bool:
    """Returns True if paddlepaddle-gpu is installed, allowed by config, and a CUDA GPU is found."""
    if not settings.ocr_use_gpu:
        logger.info("PaddleOCR: GPU disabled by configuration (ocr_use_gpu=False) — using CPU.")
        return False

    if not _HAS_TORCH:
        return False

    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            logger.info("PaddleOCR: %d GPU(s) detected — CUDA enabled.", paddle.device.cuda.device_count())
            return True
        logger.info("PaddleOCR: No CUDA GPU found — using CPU.")
        return False
    except Exception as exc:
        logger.warning("PaddleOCR: GPU detection failed (%s) — using CPU.", exc)
        return False


USE_GPU: bool = _detect_gpu()

_LANG_MAP: dict[str, str] = {
    "zh-cn": "ch",
    "zh-tw": "chinese_cht",
    "fr":    "fr",
    "de":    "german",
    "ja":    "japan",
    "ko":    "korean",
    "ar":    "ar",
    "hi":    "hi",
    "pt":    "pt",
    "ru":    "ru",
    "es":    "es",
    "it":    "it",
    "uk":    "uk",
    "vi":    "vi",
    "tr":    "tr",
    "pl":    "pl",
}


@lru_cache(maxsize=16)
def _get_engine(lang: str = "en", angle_cls: bool = False):
    """Returns a cached PaddleOCR engine. angle_cls=False for documents, True for raw images."""
    if not _HAS_PADDLEOCR:
        raise RuntimeError("PaddleOCR is not installed — OCR functionality is unavailable on this deployment.")
    logger.info("PaddleOCR: loading engine lang='%s' angle_cls=%s gpu=%s.", lang, angle_cls, USE_GPU)
    return PaddleOCR(
        use_angle_cls=angle_cls,
        lang=lang,
        use_gpu=USE_GPU,
        show_log=False,
        gpu_id=settings.ocr_gpu_id,
        gpu_mem=settings.ocr_gpu_mem,
        cpu_threads=settings.ocr_cpu_threads,
        enable_mkldnn=settings.ocr_enable_mkldnn,
    )


def _detect_language(text: str) -> str:
    """Detects the language of text and returns the matching PaddleOCR lang code. Returns 'en' on failure."""
    if not text or len(text.strip()) < 20:
        return "en"
    try:
        from langdetect import detect
        raw = detect(text)
        lang = _LANG_MAP.get(raw, "en")
        if lang != "en":
            logger.info("Language detected: '%s' → PaddleOCR lang='%s'.", raw, lang)
        return lang
    except Exception as exc:
        logger.debug("Language detection failed (%s) — defaulting to 'en'.", exc)
        return "en"


def _join_result(result) -> str:
    return "\n".join(line[1][0] for page in (result or []) if page for line in page)


def ocr_image(img_array, hint_lang: str | None = None, angle_cls: bool = False) -> str:
    """
    Runs OCR on a numpy image array.

    If hint_lang is provided, runs a single pass with that language model.
    If hint_lang is None, runs Pass 1 in English, detects the language, and
    re-runs with the correct model only if the detected language differs.

    angle_cls=False for PDF/DOCX pages (upright by design).
    angle_cls=True for raw image uploads (may be rotated).
    """
    engine = _get_engine(hint_lang or "en", angle_cls=angle_cls)
    text = _join_result(engine.ocr(img_array, cls=angle_cls))

    if hint_lang is not None:
        return text

    detected_lang = _detect_language(text)
    if detected_lang != "en":
        logger.info("Re-running OCR with lang='%s'.", detected_lang)
        second_text = _join_result(
            _get_engine(detected_lang, angle_cls=angle_cls).ocr(img_array, cls=angle_cls)
        )
        return second_text if len(second_text) >= len(text) else text

    return text
