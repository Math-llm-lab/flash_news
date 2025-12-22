from datetime import date
from pathlib import Path

from get_news import NewsConfig, build_query, make_paths


def test_build_query_includes_token_extended_name_and_crypto_terms():
    q = build_query("BTC", "Bitcoin")
    assert "btc" in q.lower()
    assert "bitcoin" in q.lower()
    assert "crypto" in q.lower()


def test_make_paths_creates_expected_filenames(tmp_path: Path):
    cfg = NewsConfig(
        tokens=["BTC"],
        news_api_key="x",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        domains="example.com",
        out_dir=tmp_path,
    )

    paths = make_paths(cfg, "BTC")
    assert paths["everything_csv"].name.endswith("_everything_BTC.csv")
    assert paths["sentiment_csv"].name.endswith("_sentiment_BTC.csv")
    assert paths["rise_json"].name.endswith("_rise_dates_BTC.json")
    assert paths["fall_json"].name.endswith("_fall_dates_BTC.json")
