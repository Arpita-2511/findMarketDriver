"""Mocked Alpaca HTTP pieces shared by Phase 4C tests (no network, fake credentials only)."""

from datetime import datetime, timezone

from services.alpaca_news_provider import AlpacaNewsProvider

KEY = "TEST-KEY-ID-not-real"
SECRET = "TEST-SECRET-not-real-9f8e7d"
FETCHED = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Returns the queued responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"params": dict(params), "headers": dict(headers)})
        if not self.responses:
            raise AssertionError("unexpected extra request")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def raw(article_id, created="2017-01-02T15:00:00Z", updated=None, symbols=("AAPL",), **extra):
    item = {"id": article_id, "headline": f"Headline {article_id}", "author": "Newsdesk",
            "created_at": created, "updated_at": updated or created, "summary": "s",
            "content": "<p>body</p>", "url": f"https://example.com/{article_id}",
            "images": [], "symbols": list(symbols), "source": "benzinga"}
    item.update(extra)
    return item


def page(items, token=None):
    return FakeResponse(payload={"news": items, "next_page_token": token})


def make_provider(session, **kwargs):
    return AlpacaNewsProvider(KEY, SECRET, session=session, sleep=lambda s: None,
                              clock=lambda: FETCHED, **kwargs)
