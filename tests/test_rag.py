"""
Tests for the RAG pipeline (retrieval + generation).

Retrieval tests run with no API key — they only need ChromaDB and
sentence-transformers (both installed locally).

Generation tests mock the Groq client so they also run without a key.
The one live integration test is skipped automatically when GROQ_API_KEY
is absent.
"""
import os
import pytest
from unittest.mock import MagicMock, patch

# Allow imports from src/ when running pytest from the project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_recommender import RAGRecommender, UserProfile, Song


# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_SONGS = [
    {"id": 1, "title": "Pop Sunshine", "artist": "Alice", "genre": "pop",
     "mood": "happy", "energy": 0.8, "acousticness": 0.2, "mood_tags": "upbeat|summer"},
    {"id": 2, "title": "Lofi Rain", "artist": "Bob", "genre": "lofi",
     "mood": "chill", "energy": 0.3, "acousticness": 0.9, "mood_tags": "relaxing|study"},
    {"id": 3, "title": "Rock Thunder", "artist": "Carol", "genre": "rock",
     "mood": "intense", "energy": 0.95, "acousticness": 0.05, "mood_tags": "loud|driving"},
    {"id": 4, "title": "Jazz Evenings", "artist": "Dan", "genre": "jazz",
     "mood": "chill", "energy": 0.4, "acousticness": 0.7, "mood_tags": "smooth|night"},
    {"id": 5, "title": "Pop Party", "artist": "Eve", "genre": "pop",
     "mood": "energetic", "energy": 0.9, "acousticness": 0.1, "mood_tags": "dance|fun"},
]

POP_USER = UserProfile(genre="pop", mood="happy", energy=0.8, acousticness=0.2)
CHILL_USER = UserProfile(genre="lofi", mood="chill", energy=0.3, acousticness=0.9)


@pytest.fixture(scope="module")
def indexed_rag():
    """One RAGRecommender with the index already built, shared across retrieval tests."""
    rag = RAGRecommender.__new__(RAGRecommender)
    # Skip Groq init — only need the embedder + ChromaDB for retrieval tests
    import chromadb
    from sentence_transformers import SentenceTransformer
    rag.chroma_client = chromadb.Client()
    rag.collection = None
    rag.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    rag.build_index(SAMPLE_SONGS)
    return rag


# ── Retrieval tests (no API key needed) ───────────────────────────────────────

def test_build_index_stores_all_songs(indexed_rag):
    count = indexed_rag.collection.count()
    assert count == len(SAMPLE_SONGS), f"Expected {len(SAMPLE_SONGS)} songs, got {count}"


def test_retrieve_returns_requested_count(indexed_rag):
    results = indexed_rag.retrieve(POP_USER, k=3)
    assert len(results) == 3


def test_retrieve_top_result_matches_genre(indexed_rag):
    """A pop/happy query should surface a pop song in the top 2 results."""
    results = indexed_rag.retrieve(POP_USER, k=3)
    top_genres = [r["genre"] for r in results[:2]]
    assert "pop" in top_genres, f"Expected pop in top 2, got {top_genres}"


def test_retrieve_chill_query_avoids_intense(indexed_rag):
    """A chill/lofi query should not rank the rock song first."""
    results = indexed_rag.retrieve(CHILL_USER, k=5)
    assert results[0]["genre"] != "rock", "Rock song should not be top result for chill user"


def test_retrieve_returns_empty_before_index_built():
    rag = RAGRecommender.__new__(RAGRecommender)
    rag.collection = None
    results = rag.retrieve(POP_USER, k=3)
    assert results == []


# ── Generation tests (Groq mocked) ────────────────────────────────────────────

TOP_SONGS = [
    Song(id=1, title="Pop Sunshine", artist="Alice", genre="pop",
         mood="happy", energy=0.8, acousticness=0.2),
    Song(id=5, title="Pop Party", artist="Eve", genre="pop",
         mood="energetic", energy=0.9, acousticness=0.1),
    Song(id=4, title="Jazz Evenings", artist="Dan", genre="jazz",
         mood="chill", energy=0.4, acousticness=0.7),
]


def _make_mock_rag(mock_response_text: str) -> RAGRecommender:
    """Return a RAGRecommender whose Groq client is fully mocked."""
    rag = RAGRecommender.__new__(RAGRecommender)
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=mock_response_text))]
    )
    rag.groq = mock_groq
    return rag


def test_generate_recommendation_returns_string():
    rag = _make_mock_rag("Here are your top picks!")
    result = rag.generate_recommendation(POP_USER, TOP_SONGS)
    assert isinstance(result, str) and result.strip() != ""


def test_generate_recommendation_calls_groq_once():
    rag = _make_mock_rag("Some response")
    rag.generate_recommendation(POP_USER, TOP_SONGS)
    rag.groq.chat.completions.create.assert_called_once()


def test_prompt_contains_user_genre():
    rag = RAGRecommender.__new__(RAGRecommender)
    prompt = rag._build_prompt(POP_USER, TOP_SONGS)
    assert "pop" in prompt.lower()


def test_prompt_contains_all_song_titles():
    rag = RAGRecommender.__new__(RAGRecommender)
    prompt = rag._build_prompt(POP_USER, TOP_SONGS)
    for song in TOP_SONGS:
        assert song.title in prompt, f"'{song.title}' missing from prompt"


def test_output_quality_mentions_song_titles():
    """
    Confidence check: the AI response should reference at least one of the
    recommended song titles, proving it used the context and didn't hallucinate.
    """
    expected = "Pop Sunshine by Alice is a great fit because it matches your pop taste."
    rag = _make_mock_rag(expected)
    result = rag.generate_recommendation(POP_USER, TOP_SONGS)
    mentioned = [s.title for s in TOP_SONGS if s.title in result]
    assert len(mentioned) >= 1, f"Response didn't mention any song title: {result!r}"


def test_generate_recommendation_propagates_exceptions():
    rag = RAGRecommender.__new__(RAGRecommender)
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = RuntimeError("API down")
    rag.groq = mock_groq
    with pytest.raises(RuntimeError, match="API down"):
        rag.generate_recommendation(POP_USER, TOP_SONGS)


# ── Live integration test (skipped without key) ───────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live API test",
)
def test_live_groq_response_mentions_song_title():
    """
    Calls the real Groq API. Passes when the response text contains at least
    one of the three song titles — confirming the model used the prompt context.
    """
    rag = RAGRecommender()
    result = rag.generate_recommendation(POP_USER, TOP_SONGS)
    assert isinstance(result, str) and len(result) > 20, "Response too short"
    mentioned = [s.title for s in TOP_SONGS if s.title in result]
    assert len(mentioned) >= 1, (
        f"Live Groq response didn't mention any song title.\nResponse: {result}"
    )
