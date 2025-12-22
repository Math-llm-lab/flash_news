"""flash_news.get_news

Public demo extracted from a larger private research system (under NDA).

This module fetches crypto-related news via NewsAPI, then (optionally) asks an
LLM to label each article as:
  - historical: True/False (mostly past events vs. current catalyst)
  - sentiment: one of a small, fixed taxonomy

Outputs are written under ``data/news/<TOKEN>/`` as CSV/JSON so a reviewer can
inspect results without needing access to any private infrastructure.

Design goals for this public repo:
  - runnable in <10 minutes
  - testable without network calls
  - clear boundaries (HTTP client vs. labeling vs. IO)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

SENTIMENT_OPTIONS: Tuple[str, ...] = (
    "significant rise",
    "rise",
    "neutral",
    "fall",
    "significant fall",
)


@dataclass(frozen=True)
class NewsConfig:
    """Configuration for one fetch + label run."""

    tokens: Sequence[str]
    news_api_key: str
    start_date: date
    end_date: date
    domains: str
    language: str = "en"
    out_dir: Path = Path("data/news")
    request_timeout_s: int = 25
    request_sleep_s: float = 0.8  # basic rate-limit friendliness

    def validate(self) -> None:
        if not self.tokens:
            raise ValueError("tokens must be non-empty")
        if not self.news_api_key:
            raise ValueError("news_api_key must be set")
        if self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")


class LlmClient(Protocol):
    """Minimal protocol so we can unit-test without importing the OpenAI SDK."""

    def label(self, token: str, news_text: str) -> Tuple[Optional[bool], Optional[str]]:
        """Return (historical, sentiment)."""


def build_query(token: str, extended_name: Optional[str] = None) -> str:
    """Build a NewsAPI query string."""
    token_clean = token.strip()
    parts = [token_clean, token_clean.lower()]

    if extended_name:
        en = extended_name.strip()
        if en and en.lower() != token_clean.lower():
            parts.extend([en, en.lower()])

    base = " OR ".join(dict.fromkeys(parts))
    return f"({base}) AND (crypto OR cryptocurrency OR blockchain)"


def make_paths(cfg: NewsConfig, token: str) -> Dict[str, Path]:
    """Standardize output file names/paths."""
    token_dir = cfg.out_dir / token.upper()
    token_dir.mkdir(parents=True, exist_ok=True)

    period = f"{cfg.start_date:%Y-%m-%d}_to_{cfg.end_date:%Y-%m-%d}"
    return {
        "everything_csv": token_dir / f"{period}_everything_{token.upper()}.csv",
        "sentiment_csv": token_dir / f"{period}_sentiment_{token.upper()}.csv",
        "rise_json": token_dir / f"{period}_rise_dates_{token.upper()}.json",
        "fall_json": token_dir / f"{period}_fall_dates_{token.upper()}.json",
    }


def _safe_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_bool(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if re.search(r"\btrue\b", t):
        return True
    if re.search(r"\bfalse\b", t):
        return False
    return None


def parse_llm_output(text: str) -> Tuple[Optional[bool], Optional[str]]:
    """Parse an LLM response robustly."""
    t = (text or "").strip().lower()

    is_hist = _normalize_bool(t)

    sentiment = None
    for opt in sorted(SENTIMENT_OPTIONS, key=len, reverse=True):
        if opt in t:
            sentiment = opt
            break

    return is_hist, sentiment


class NewsApiClient:
    """Thin wrapper around NewsAPI HTTP calls (easy to mock)."""

    def __init__(self, api_key: str, *, timeout_s: int = 25):
        if not api_key:
            raise ValueError("api_key must be set")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def everything(
        self,
        *,
        query: str,
        start_date: date,
        end_date: date,
        domains: str,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        encoded_q = urllib.parse.quote(query)

        url = (
            "https://newsapi.org/v2/everything"
            f"?q={encoded_q}"
            f"&from={start_date:%Y-%m-%d}"
            f"&to={end_date:%Y-%m-%d}"
            f"&domains={urllib.parse.quote(domains)}"
            f"&language={language}"
            f"&apiKey={self.api_key}"
        )

        resp = requests.get(url, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"NewsAPI error {resp.status_code}: {resp.text[:400]}")

        payload = resp.json() if resp.content else {}
        articles = payload.get("articles", []) or []
        if not isinstance(articles, list):
            raise RuntimeError("Unexpected NewsAPI response shape: 'articles' is not a list")
        return articles


class OpenAiChatLabeler:
    """OpenAI chat-completions labeler."""

    def __init__(self, client: Any, *, model: str):
        self._client = client
        self._model = model

    def label(self, token: str, news_text: str) -> Tuple[Optional[bool], Optional[str]]:
        prompt = (
            f"You evaluate crypto news impact for {token}.\n\n"
            "Return exactly two lines:\n"
            "1) historical: True or False (True = mostly past events)\n"
            f"2) sentiment: one of {', '.join(SENTIMENT_OPTIONS)}\n\n"
            "News:\n"
            f"{news_text}\n"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return parse_llm_output(text)


class CryptoNewsAnalyzer:
    def __init__(
        self,
        cfg: NewsConfig,
        *,
        news_client: Optional[NewsApiClient] = None,
        llm_client: Optional[LlmClient] = None,
        extended_names: Optional[Mapping[str, str]] = None,
    ):
        cfg.validate()
        self.cfg = cfg
        self.news_client = news_client or NewsApiClient(
            cfg.news_api_key, timeout_s=cfg.request_timeout_s
        )
        self.llm_client = llm_client
        self.news_data: Dict[str, List[Dict[str, Any]]] = {t.upper(): [] for t in cfg.tokens}
        self.extended_names: Dict[str, str] = {"BTC": "Bitcoin", "ETH": "Ethereum"}
        if extended_names:
            self.extended_names.update({k.upper(): v for k, v in extended_names.items()})

    def fetch_news(self, token: str) -> List[Dict[str, Any]]:
        """Fetch and cache articles for a token."""
        token = token.upper()
        paths = make_paths(self.cfg, token)

        if paths["everything_csv"].exists():
            df = pd.read_csv(paths["everything_csv"])
            articles = [row.dropna().to_dict() for _, row in df.iterrows()]
            self.news_data[token] = [{"news": a, "top_headlines": False} for a in articles]
            LOGGER.info("Loaded cached NewsAPI results for %s (%d rows)", token, len(articles))
            return articles

        query = build_query(token, self.extended_names.get(token))
        LOGGER.info("Fetching NewsAPI 'everything' for %s", token)
        articles = self.news_client.everything(
            query=query,
            start_date=self.cfg.start_date,
            end_date=self.cfg.end_date,
            domains=self.cfg.domains,
            language=self.cfg.language,
        )

        if articles:
            pd.DataFrame(articles).to_csv(paths["everything_csv"], index=False)
        self.news_data[token] = [{"news": a, "top_headlines": False} for a in articles]

        time.sleep(self.cfg.request_sleep_s)
        return articles

    def analyze_sentiment(self) -> None:
        """Attach LLM labels and write a compact CSV."""
        if not self.llm_client:
            raise RuntimeError("LLM client not provided. Pass llm_client to analyze sentiment.")

        for token, wrapped_articles in self.news_data.items():
            paths = make_paths(self.cfg, token)

            if paths["sentiment_csv"].exists():
                df = pd.read_csv(paths["sentiment_csv"])
                for article, (_, row) in zip(wrapped_articles, df.iterrows(), strict=False):
                    article["is_historical"] = bool(row.get("is_historical"))
                    article["sentiment"] = str(row.get("sentiment"))
                LOGGER.info("Loaded cached sentiment for %s (%d rows)", token, len(df))
                continue

            results_rows: List[Dict[str, Any]] = []
            for article in wrapped_articles:
                news = article.get("news") or {}
                title = news.get("title") or ""
                desc = news.get("description") or ""
                content = news.get("content") or ""
                news_text = f"{title}\n{desc}\n{content}".strip()

                is_hist, sentiment = self.llm_client.label(token, news_text)

                if is_hist is None:
                    is_hist = False
                if sentiment is None:
                    sentiment = "neutral"

                article["is_historical"] = is_hist
                article["sentiment"] = sentiment

                results_rows.append(
                    {
                        "publishedAt": news.get("publishedAt"),
                        "title": title,
                        "url": news.get("url"),
                        "is_historical": is_hist,
                        "sentiment": sentiment,
                    }
                )

            if results_rows:
                pd.DataFrame(results_rows).to_csv(paths["sentiment_csv"], index=False)

    def summarize_events(self) -> Dict[str, Dict[str, Any]]:
        """Summarize rise/fall signals and write JSON date lists."""
        out: Dict[str, Dict[str, Any]] = {}

        for token, articles in self.news_data.items():
            paths = make_paths(self.cfg, token)
            fresh = [a for a in articles if a.get("is_historical") is False]

            rise = sorted(
                [a for a in fresh if a.get("sentiment") == "significant rise"],
                key=lambda x: (x.get("news") or {}).get("publishedAt") or "",
            )
            fall = sorted(
                [a for a in fresh if a.get("sentiment") in ("fall", "significant fall")],
                key=lambda x: (x.get("news") or {}).get("publishedAt") or "",
            )

            rise_dates = [
                (a.get("news") or {}).get("publishedAt")
                for a in rise
                if (a.get("news") or {}).get("publishedAt")
            ]
            fall_dates = [
                (a.get("news") or {}).get("publishedAt")
                for a in fall
                if (a.get("news") or {}).get("publishedAt")
            ]

            _safe_json_write(paths["rise_json"], rise_dates)
            _safe_json_write(paths["fall_json"], fall_dates)

            out[token] = {
                "total_articles": len(articles),
                "fresh_articles": len(fresh),
                "rise_count": len(rise_dates),
                "fall_count": len(fall_dates),
                "rise_dates": rise_dates,
                "fall_dates": fall_dates,
            }

        return out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flash News: fetch crypto news + optional LLM labeling")
    p.add_argument("--tokens", default="BTC,ETH,LINK", help="Comma-separated token symbols")
    p.add_argument("--start", default="2024-09-27", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-10-06", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--domains",
        default=(
            "coindesk.com,cointelegraph.com,decrypt.co,cryptoslate.com,bitcoinmagazine.com,"
            "newsbtc.com,cryptobriefing.com,theblock.co,ambcrypto.com"
        ),
        help="Comma-separated domain allowlist",
    )
    p.add_argument("--out-dir", default="data/news", help="Output directory")
    p.add_argument("--language", default="en", help="Language code")
    p.add_argument("--label", action="store_true", help="Run LLM labeling (requires OPENAI_API_KEY)")
    p.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING...")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    load_dotenv()
    args = _parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        raise SystemExit("Missing NEWS_API_KEY. Put it in .env or environment variables.")

    tokens = [t.strip().upper() for t in str(args.tokens).split(",") if t.strip()]
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    cfg = NewsConfig(
        tokens=tokens,
        news_api_key=news_api_key,
        start_date=start,
        end_date=end,
        domains=str(args.domains),
        language=str(args.language),
        out_dir=Path(str(args.out_dir)),
    )

    llm_client: Optional[LlmClient] = None
    if args.label:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise SystemExit("Missing OPENAI_API_KEY. Put it in .env or environment variables.")

        from openai import OpenAI  # lazy import

        client = OpenAI(api_key=openai_api_key)
        llm_client = OpenAiChatLabeler(client, model=str(args.openai_model))

    analyzer = CryptoNewsAnalyzer(cfg, llm_client=llm_client)

    for t in cfg.tokens:
        analyzer.fetch_news(t)

    if args.label:
        analyzer.analyze_sentiment()

    summary = analyzer.summarize_events()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
