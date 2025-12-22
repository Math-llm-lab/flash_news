# get_news.py
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


# -----------------------------
# Configuration / helpers
# -----------------------------

SENTIMENT_OPTIONS = ("significant rise", "rise", "neutral", "fall", "significant fall")


@dataclass(frozen=True)
class NewsConfig:
    tokens: List[str]
    news_api_key: str
    start_date: datetime
    end_date: datetime
    domains: str
    language: str = "en"
    out_dir: Path = Path("data/news")
    request_timeout_s: int = 25
    request_sleep_s: float = 1.0  # basic rate-limit friendliness


def build_query(token: str, extended_name: Optional[str] = None) -> str:
    """
    Build a NewsAPI query string. Keep it simple and explicit.
    """
    token_clean = token.strip()
    parts = [token_clean, token_clean.lower()]

    if extended_name:
        en = extended_name.strip()
        if en and en.lower() != token_clean.lower():
            parts.extend([en, en.lower()])

    # Keep broad crypto keywords, but don't overdo it (better precision).
    base = " OR ".join(dict.fromkeys(parts))  # de-dup, preserve order
    return f"({base}) AND (crypto OR cryptocurrency OR blockchain)"


def make_paths(cfg: NewsConfig, token: str) -> Dict[str, Path]:
    """
    Standardize output file names/paths.
    """
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
    """
    Parse LLM output robustly, even if formatting is messy.

    Returns:
      (is_historical, sentiment)
    """
    t = (text or "").strip().lower()

    # historical: search true/false anywhere
    is_hist = _normalize_bool(t)

    # sentiment: pick first match in options (longest first helps)
    sentiment = None
    for opt in sorted(SENTIMENT_OPTIONS, key=len, reverse=True):
        if opt in t:
            sentiment = opt
            break

    return is_hist, sentiment


# -----------------------------
# Main class
# -----------------------------

class CryptoNewsAnalyzer:
    def __init__(self, cfg: NewsConfig, openai_client: Optional[Any] = None, openai_model: Optional[str] = None):
        self.cfg = cfg
        self.openai_client = openai_client
        self.openai_model = openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Keep domains list as a single comma-separated string for NewsAPI
        self.news_data: Dict[str, List[Dict[str, Any]]] = {t.upper(): [] for t in cfg.tokens}

        # Extended names mapping (expand if needed)
        self.extended_names = {"BTC": "Bitcoin", "ETH": "Ethereum"}

    def fetch_news(self, token: str) -> None:
        token = token.upper()
        paths = make_paths(self.cfg, token)

        if paths["everything_csv"].exists():
            df = pd.read_csv(paths["everything_csv"])
            self.news_data[token] = [{"news": row.to_dict(), "top_headlines": False} for _, row in df.iterrows()]
            return

        query = build_query(token, self.extended_names.get(token))
        encoded_q = urllib.parse.quote(query)

        url = (
            "https://newsapi.org/v2/everything"
            f"?q={encoded_q}"
            f"&from={self.cfg.start_date:%Y-%m-%d}"
            f"&to={self.cfg.end_date:%Y-%m-%d}"
            f"&domains={urllib.parse.quote(self.cfg.domains)}"
            f"&language={self.cfg.language}"
            f"&apiKey={self.cfg.news_api_key}"
        )

        resp = requests.get(url, timeout=self.cfg.request_timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"NewsAPI error {resp.status_code}: {resp.text[:400]}")

        payload = resp.json()
        articles = payload.get("articles", []) or []

        if articles:
            pd.DataFrame(articles).to_csv(paths["everything_csv"], index=False)

        self.news_data[token] = [{"news": a, "top_headlines": False} for a in articles]

        # Small sleep to be polite with APIs
        time.sleep(self.cfg.request_sleep_s)

    def analyze_sentiment(self) -> None:
        """
        Adds:
          article['is_historical']: bool
          article['sentiment']: one of SENTIMENT_OPTIONS
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client not provided. Provide openai_client to analyze sentiment.")

        for token, articles in self.news_data.items():
            paths = make_paths(self.cfg, token)

            # If already computed, load and attach
            if paths["sentiment_csv"].exists():
                df = pd.read_csv(paths["sentiment_csv"])
                for article, (_, row) in zip(articles, df.iterrows()):
                    article["is_historical"] = bool(row.get("is_historical"))
                    article["sentiment"] = str(row.get("sentiment"))
                continue

            results_rows = []
            for article in articles:
                title = (article["news"] or {}).get("title") or ""
                desc = (article["news"] or {}).get("description") or ""
                content = (article["news"] or {}).get("content") or ""
                news_text = f"{title}\n{desc}\n{content}".strip()

                is_hist, sentiment = self._llm_sentiment(token, news_text)

                # fallbacks (avoid None in CSV)
                if is_hist is None:
                    is_hist = False
                if sentiment is None:
                    sentiment = "neutral"

                article["is_historical"] = is_hist
                article["sentiment"] = sentiment

                results_rows.append({
                    "publishedAt": (article["news"] or {}).get("publishedAt"),
                    "title": title,
                    "url": (article["news"] or {}).get("url"),
                    "is_historical": is_hist,
                    "sentiment": sentiment,
                })

            if results_rows:
                pd.DataFrame(results_rows).to_csv(paths["sentiment_csv"], index=False)

    def _llm_sentiment(self, token: str, news_text: str) -> Tuple[Optional[bool], Optional[str]]:
        prompt = (
            f"You evaluate crypto news impact for {token}.\n\n"
            "Return exactly two lines:\n"
            "1) historical: True or False (True = mostly past events)\n"
            f"2) sentiment: one of {', '.join(SENTIMENT_OPTIONS)}\n\n"
            "News:\n"
            f"{news_text}\n"
        )

        # Compatible with OpenAI Python SDK v1 style client
        resp = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return parse_llm_output(text)

    def summarize_events(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns summary per token and writes rise/fall dates JSON.
        """
        out: Dict[str, Dict[str, Any]] = {}

        for token, articles in self.news_data.items():
            paths = make_paths(self.cfg, token)

            # Non-historical only
            fresh = [a for a in articles if a.get("is_historical") is False]

            rise = sorted(
                [a for a in fresh if a.get("sentiment") == "significant rise"],
                key=lambda x: (x.get("news") or {}).get("publishedAt") or "",
            )
            fall = sorted(
                [a for a in fresh if a.get("sentiment") in ("fall", "significant fall")],
                key=lambda x: (x.get("news") or {}).get("publishedAt") or "",
            )

            rise_dates = [(a.get("news") or {}).get("publishedAt") for a in rise if (a.get("news") or {}).get("publishedAt")]
            fall_dates = [(a.get("news") or {}).get("publishedAt") for a in fall if (a.get("news") or {}).get("publishedAt")]

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


# -----------------------------
# CLI usage (simple demo)
# -----------------------------

def main() -> None:
    load_dotenv()  # reads .env if present

    news_api_key = os.getenv("NEWS_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not news_api_key:
        raise SystemExit("Missing NEWS_API_KEY. Put it in .env or environment variables.")
    if not openai_api_key:
        raise SystemExit("Missing OPENAI_API_KEY. Put it in .env or environment variables.")

    # Lazy import so tests don't require openai installed
    from openai import OpenAI

    client = OpenAI(api_key=openai_api_key)

    cfg = NewsConfig(
        tokens=["BTC", "ETH", "LINK"],
        news_api_key=news_api_key,
        start_date=datetime(2024, 9, 27),
        end_date=datetime(2024, 10, 6),
        domains="coindesk.com,cointelegraph.com,decrypt.co,cryptoslate.com,bitcoinmagazine.com,newsbtc.com,cryptobriefing.com,theblock.co,ambcrypto.com",
    )

    analyzer = CryptoNewsAnalyzer(cfg, openai_client=client)

    for t in cfg.tokens:
        analyzer.fetch_news(t)

    analyzer.analyze_sentiment()
    summary = analyzer.summarize_events()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
