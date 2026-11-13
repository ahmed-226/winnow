"""Opt-in live test against the real Gemini embedding API.

Run only when GEMINI_API_KEY is exported in the environment:

    $env:GEMINI_API_KEY="..."; uv run pytest tests/live -q

Never runs in CI or by default; the marker also allows `-m "not live"` selection.
"""

from __future__ import annotations

import math
import os

import pytest

from app.config import Settings
from app.services.embeddings import build_embedding_provider

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="opt-in: requires GEMINI_API_KEY in the environment",
    ),
]


def live_settings() -> Settings:
    return Settings(
        embedding_provider="gemini",
        gemini_api_key=os.environ["GEMINI_API_KEY"],
    )  # type: ignore[call-arg]


async def test_live_query_embedding() -> None:
    provider = build_embedding_provider(live_settings())
    vector = await provider.embed_query("How many weeks of parental leave do I get?")
    assert len(vector) == 1536
    assert all(math.isfinite(value) for value in vector)
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-6)


async def test_live_document_embeddings_differ_by_text() -> None:
    provider = build_embedding_provider(live_settings())
    vectors = await provider.embed_documents(
        ["Parental leave is sixteen weeks.", "The quarterly report shows rising costs."]
    )
    assert len(vectors) == 2
    assert all(len(vector) == 1536 for vector in vectors)
    assert vectors[0] != vectors[1]
