from datetime import date

import pytest

from get_news import NewsApiClient


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.content = b"x"

    def json(self):
        return self._payload


def test_newsapi_client_raises_on_non_200(monkeypatch):
    def fake_get(url, timeout):
        return _Resp(401, text="nope")

    monkeypatch.setattr("requests.get", fake_get)
    client = NewsApiClient("k")
    with pytest.raises(RuntimeError):
        client.everything(
            query="(BTC) AND crypto",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            domains="example.com",
        )


def test_newsapi_client_returns_articles_list(monkeypatch):
    def fake_get(url, timeout):
        return _Resp(200, payload={"articles": [{"title": "t"}]})

    monkeypatch.setattr("requests.get", fake_get)
    client = NewsApiClient("k")
    articles = client.everything(
        query="(BTC) AND crypto",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 2),
        domains="example.com",
    )
    assert articles == [{"title": "t"}]


def test_newsapi_client_rejects_unexpected_articles_shape(monkeypatch):
    def fake_get(url, timeout):
        return _Resp(200, payload={"articles": {"not": "a list"}})

    monkeypatch.setattr("requests.get", fake_get)
    client = NewsApiClient("k")
    with pytest.raises(RuntimeError):
        client.everything(
            query="(BTC) AND crypto",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            domains="example.com",
        )
