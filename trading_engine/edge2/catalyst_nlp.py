"""
Layer 3: NLP Catalyst Engine
============================
Classifies real-time news headlines into High-Velocity Catalysts vs.
Low-Velocity/Noise and assigns a normalized Catalyst Score [0.0, 1.0].

Design notes:
- Hybrid architecture: fast regex/keyword category gating + FinBERT sentiment.
  Pure-transformer classification of catalyst *category* is slow and
  unnecessary; category is a vocabulary problem, sentiment is not.
- Fully async-compatible: model inference runs in a thread executor so it
  never blocks the BreakoutScanner event loop.
- Graceful degradation: if transformers/torch are unavailable or the model
  fails to load (e.g., low-RAM Replit), falls back to keyword-only scoring.

Dependencies (requirements.txt):
    transformers>=4.40.0
    torch>=2.2.0          # CPU build is fine: pip install torch --index-url https://download.pytorch.org/whl/cpu
    # No other deps; stdlib otherwise.

Usage:
    processor = CatalystNLPProcessor()
    result = await processor.process("XYZ receives FDA approval for ...", ticker="XYZ")
    if result.is_high_velocity and result.catalyst_score >= 0.65:
        scanner.flag_catalyst(result)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

logger = logging.getLogger("catalyst_nlp")


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

class CatalystCategory(str, Enum):
    FDA_REGULATORY = "fda_regulatory"
    MERGER_ACQUISITION = "merger_acquisition"
    CONTRACT_PARTNERSHIP = "contract_partnership"
    EARNINGS_GUIDANCE = "earnings_guidance"
    SHORT_SQUEEZE_STRUCTURE = "short_squeeze_structure"
    DILUTION_OFFERING = "dilution_offering"      # negative catalyst — kills low-float longs
    PROMOTIONAL_FLUFF = "promotional_fluff"
    MACRO_GENERAL = "macro_general"
    UNCLASSIFIED = "unclassified"


class Velocity(str, Enum):
    HIGH = "high_velocity"
    LOW = "low_velocity_noise"
    NEGATIVE = "negative_catalyst"   # actively bearish for breakout longs


@dataclass(frozen=True)
class CatalystResult:
    text: str
    ticker: Optional[str]
    category: CatalystCategory
    velocity: Velocity
    sentiment_label: str            # "positive" | "negative" | "neutral" | "unavailable"
    sentiment_score: float          # FinBERT confidence in label, 0-1
    catalyst_score: float           # normalized composite, 0.0-1.0
    latency_ms: float
    model_used: bool                # False => keyword-only fallback path
    matched_patterns: tuple = field(default_factory=tuple)

    @property
    def is_high_velocity(self) -> bool:
        return self.velocity == Velocity.HIGH

    @property
    def is_negative(self) -> bool:
        return self.velocity == Velocity.NEGATIVE


# --------------------------------------------------------------------------
# Category vocabulary
# --------------------------------------------------------------------------
# Each entry: (compiled_pattern, category, base_weight)
# base_weight reflects historical breakout velocity of the catalyst class.

_CATEGORY_PATTERNS: list[tuple[re.Pattern, CatalystCategory, float]] = [
    # --- High velocity ---
    (re.compile(r"\b(fda (approval|clearance|fast.?track|breakthrough)|510\(k\)|nda accept|pdufa|phase (2|3|ii|iii) (data|results|success|met))\b", re.I),
        CatalystCategory.FDA_REGULATORY, 0.95),
    (re.compile(r"\b(merger|acquisition|acquires?|to be acquired|buyout|takeover|tender offer|definitive agreement)\b", re.I),
        CatalystCategory.MERGER_ACQUISITION, 0.90),
    (re.compile(r"\b((multi.?year|enterprise|government|defense|doD|nasa) (contract|deal|order)|contract award|purchase order worth|\$\d+(\.\d+)?\s?(m|b|million|billion) (contract|order|deal))\b", re.I),
        CatalystCategory.CONTRACT_PARTNERSHIP, 0.85),
    (re.compile(r"\b(partnership with (amazon|microsoft|google|apple|nvidia|tesla|openai|meta)|strategic (partnership|collaboration|investment))\b", re.I),
        CatalystCategory.CONTRACT_PARTNERSHIP, 0.80),
    (re.compile(r"\b(record (revenue|earnings|quarter)|beats? (estimates|expectations)|raises? (guidance|outlook)|profitability ahead of schedule)\b", re.I),
        CatalystCategory.EARNINGS_GUIDANCE, 0.75),
    (re.compile(r"\b(short squeeze|short interest (above|exceeds|over)|low float|reverse split effective|uplist(ing)? to (nasdaq|nyse))\b", re.I),
        CatalystCategory.SHORT_SQUEEZE_STRUCTURE, 0.70),

    # --- Negative catalysts (structurally bearish for longs) ---
    (re.compile(r"\b(registered direct|public offering|private placement|at.?the.?market|atm (program|offering)|dilut(ion|ive)|warrant (exercise|inducement)|s-3|s-1 filing|going concern|delisting notice|sec (investigation|subpoena)|clinical hold|crl|complete response letter|trial (halt|failure|miss))\b", re.I),
        CatalystCategory.DILUTION_OFFERING, 0.85),

    # --- Low velocity / noise ---
    (re.compile(r"\b(to (present|attend|participate) at .*(conference|summit)|investor (awareness|relations) (campaign|program)|featured (article|interview)|stock (alert|watch|pick)|why .* (stock|shares) (is|are) (up|down|moving)|top \d+ stocks)\b", re.I),
        CatalystCategory.PROMOTIONAL_FLUFF, 0.10),
    (re.compile(r"\b(fed(eral reserve)?|fomc|cpi|jobs report|treasury yields?|market (recap|wrap)|futures (rise|fall|point))\b", re.I),
        CatalystCategory.MACRO_GENERAL, 0.05),
]

_HIGH_VELOCITY_CATEGORIES = {
    CatalystCategory.FDA_REGULATORY,
    CatalystCategory.MERGER_ACQUISITION,
    CatalystCategory.CONTRACT_PARTNERSHIP,
    CatalystCategory.EARNINGS_GUIDANCE,
    CatalystCategory.SHORT_SQUEEZE_STRUCTURE,
}


# --------------------------------------------------------------------------
# Processor
# --------------------------------------------------------------------------

class CatalystNLPProcessor:
    """
    Async-friendly NLP catalyst classifier.

    Parameters
    ----------
    model_name : HF model id for financial sentiment. Default: ProsusAI/finbert.
    enable_model : set False to force keyword-only mode (e.g., on Replit).
    max_workers : executor threads for inference. 1 is correct for a single
                  CPU-bound model; raise only if you shard models.
    sentiment_weight : how much FinBERT sentiment modulates the category
                       base weight (0 = ignore sentiment entirely).
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        enable_model: bool = True,
        max_workers: int = 1,
        sentiment_weight: float = 0.35,
    ):
        self.model_name = model_name
        self.sentiment_weight = max(0.0, min(1.0, sentiment_weight))
        self._pipeline = None
        self._model_failed = False
        self._enable_model = enable_model
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="finbert")
        self._load_lock = asyncio.Lock()

    # ---------------- model lifecycle ----------------

    async def _ensure_model(self) -> bool:
        """Lazy-load FinBERT once, off the event loop. Returns availability."""
        if not self._enable_model or self._model_failed:
            return False
        if self._pipeline is not None:
            return True
        async with self._load_lock:
            if self._pipeline is not None:          # double-checked
                return True
            loop = asyncio.get_running_loop()
            try:
                self._pipeline = await loop.run_in_executor(
                    self._executor, self._load_pipeline_sync
                )
                logger.info("FinBERT loaded: %s", self.model_name)
                return True
            except Exception as exc:                # noqa: BLE001
                self._model_failed = True
                logger.warning(
                    "FinBERT unavailable (%s). Falling back to keyword-only "
                    "scoring. Install torch+transformers or set "
                    "enable_model=False to silence this.", exc
                )
                return False

    def _load_pipeline_sync(self):
        from transformers import pipeline  # deferred import
        return pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=self.model_name,
            truncation=True,
            max_length=256,
            device=-1,  # CPU; set 0 / "mps" manually if you want GPU/Apple Silicon
        )

    # ---------------- classification core ----------------

    @staticmethod
    def _classify_category(text: str) -> tuple[CatalystCategory, float, tuple]:
        """First-match-wins over priority-ordered patterns."""
        matched = []
        best: tuple[CatalystCategory, float] | None = None
        for pattern, category, weight in _CATEGORY_PATTERNS:
            if pattern.search(text):
                matched.append(category.value)
                if best is None or weight > best[1]:
                    best = (category, weight)
        if best is None:
            return CatalystCategory.UNCLASSIFIED, 0.30, tuple(matched)
        return best[0], best[1], tuple(matched)

    def _run_sentiment_sync(self, text: str) -> tuple[str, float]:
        out = self._pipeline(text[:512])[0]
        return out["label"].lower(), float(out["score"])

    @staticmethod
    def _compose_score(
        base_weight: float,
        sentiment_label: str,
        sentiment_conf: float,
        sentiment_weight: float,
        category: CatalystCategory,
    ) -> float:
        """
        score = base * (1 + sw * sentiment_adjustment), clamped to [0, 1].
        Dilution/negative categories are scored on magnitude but surfaced
        via Velocity.NEGATIVE — the scanner must treat them as avoid signals.
        """
        adj = 0.0
        if sentiment_label == "positive":
            adj = sentiment_conf
        elif sentiment_label == "negative":
            adj = -sentiment_conf
        # neutral / unavailable => 0

        if category == CatalystCategory.DILUTION_OFFERING:
            # Negative news + negative sentiment = strong avoid signal.
            adj = -adj  # invert: more negative sentiment => higher (avoid) score

        score = base_weight * (1.0 + sentiment_weight * adj)
        return round(max(0.0, min(1.0, score)), 4)

    # ---------------- public API ----------------

    async def process(self, text: str, ticker: Optional[str] = None) -> CatalystResult:
        """Classify a single headline. Never raises; errors degrade to fallback."""
        start = time.perf_counter()

        if not text or not text.strip():
            return CatalystResult(
                text=text or "", ticker=ticker,
                category=CatalystCategory.UNCLASSIFIED,
                velocity=Velocity.LOW,
                sentiment_label="unavailable", sentiment_score=0.0,
                catalyst_score=0.0,
                latency_ms=0.0, model_used=False,
            )

        text = text.strip()
        category, base_weight, matched = self._classify_category(text)

        sentiment_label, sentiment_conf, model_used = "unavailable", 0.0, False
        if await self._ensure_model():
            loop = asyncio.get_running_loop()
            try:
                sentiment_label, sentiment_conf = await loop.run_in_executor(
                    self._executor, self._run_sentiment_sync, text
                )
                model_used = True
            except Exception as exc:                # noqa: BLE001
                logger.error("Sentiment inference failed: %s", exc)

        score = self._compose_score(
            base_weight, sentiment_label, sentiment_conf,
            self.sentiment_weight, category,
        )

        if category == CatalystCategory.DILUTION_OFFERING:
            velocity = Velocity.NEGATIVE
        elif category in _HIGH_VELOCITY_CATEGORIES and score >= 0.50:
            velocity = Velocity.HIGH
        else:
            velocity = Velocity.LOW

        return CatalystResult(
            text=text, ticker=ticker, category=category, velocity=velocity,
            sentiment_label=sentiment_label, sentiment_score=round(sentiment_conf, 4),
            catalyst_score=score,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            model_used=model_used, matched_patterns=matched,
        )

    async def process_batch(
        self, items: Iterable[tuple[str, Optional[str]]], concurrency: int = 8
    ) -> list[CatalystResult]:
        """
        Batch interface: items = [(headline, ticker), ...].
        Concurrency bounds the regex/dispatch fan-out; actual model inference
        is serialized by the single-thread executor, which is intentional.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _one(text: str, ticker: Optional[str]) -> CatalystResult:
            async with sem:
                return await self.process(text, ticker)

        return await asyncio.gather(*(_one(t, tk) for t, tk in items))

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    SAMPLES = [
        ("ABCD receives FDA approval for lead oncology candidate", "ABCD"),
        ("WXYZ announces $45 million multi-year defense contract award", "WXYZ"),
        ("QRST to present at the Emerging Growth Conference next week", "QRST"),
        ("LMNO announces $10M registered direct offering priced at-the-market", "LMNO"),
        ("Fed officials signal rates to stay higher for longer", None),
        ("EFGH enters strategic partnership with NVIDIA for edge AI", "EFGH"),
    ]

    async def main():
        proc = CatalystNLPProcessor(enable_model=("--no-model" not in __import__("sys").argv))
        results = await proc.process_batch(SAMPLES)
        for r in results:
            print(f"[{r.velocity.value:>18}] score={r.catalyst_score:.2f} "
                  f"cat={r.category.value:<24} sent={r.sentiment_label:<11} "
                  f"({r.latency_ms}ms, model={r.model_used}) :: {r.text[:60]}")
        await proc.aclose()

    asyncio.run(main())
