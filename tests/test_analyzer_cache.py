from datetime import date

import pandas as pd

from get_news import CryptoNewsAnalyzer, NewsApiClient, NewsConfig


class _FakeNewsClient(NewsApiClient):
    def __init__(self, articles):
        self._articles = articles
        self.calls = 0

    def everything(self, *, query, start_date, end_date, domains, language="en"):
        self.calls += 1
        return self._articles


def test_fetch_news_writes_csv_then_uses_cache(tmp_path):
    cfg = NewsConfig(
        tokens=["BTC"],
        news_api_key="k",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        domains="example.com",
        out_dir=tmp_path,
        request_sleep_s=0,
    )

    articles = [{"title": "hello", "url": "u"}]
    client = _FakeNewsClient(articles)

    a = CryptoNewsAnalyzer(cfg, news_client=client, llm_client=None)
    out1 = a.fetch_news("BTC")
    assert out1 == articles
    assert client.calls == 1

    out2 = a.fetch_news("BTC")
    assert client.calls == 1
    assert isinstance(out2, list) and out2[0]["title"] == "hello"

    token_dir = tmp_path / "BTC"
    cached = list(token_dir.glob("*_everything_BTC.csv"))
    assert cached, "expected cached CSV"
    df = pd.read_csv(cached[0])
    assert df.iloc[0]["title"] == "hello"
