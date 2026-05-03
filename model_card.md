# 🎧 Model Card: Music Recommender (updated to match code)

This model card summarizes the implementation in `src/recommender.py` and `src/rag_recommender.py` and documents the dataset fields, scoring behavior, retrieval/RAG setup, strengths, limitations, and evaluation notes.

## 1. Model Name

Music Recommender Simulation

---

## 2. Intended Use

This system produces personalized song suggestions for interactive demos and classroom exploration. It is intended to help users discover tracks by combining simple preference matching (genre, mood, energy) with retrieval-augmented explanations. It assumes user-provided preferences (genre, mood, numeric energy/acousticness, and optional lists of liked/skipped songs). This repository is a demonstration and not a production-ready recommender.

---

## 3. How the Model Works (code-accurate)

- Core scoring: implemented in `score_song()` in `src/recommender.py`. The default scoring weights are:

	- `genre`: 0.30
	- `mood`: 0.25
	- `energy`: 0.25
	- `acousticness`: 0.15
	- `feedback` (previous likes/ skips): 0.05

- Ranking strategies: `RANKING_STRATEGIES` defines three pre-set weight distributions (`balanced`, `genre_first`, `mood_first`) that the functional `recommend_songs()` can use instead of the default weights. These strategies include additional dimensions such as `popularity`, `decade`, and `mood_tags` for alternate ranking behavior.

- Feature handling:
	- Song numeric fields in CSV are normalized (0–1 in CSV → 1–10 used for scoring for energy and acousticness).
	- Genre and mood matching are binary (10 for exact match, 0 otherwise).
	- Feedback: exact title matches or id-based likes/skips are used to boost or penalize scores (skips subtract a small penalty).

- Diversity re-ranking: after scoring, `diversity_rerank()` greedily enforces diversity constraints with defaults `max_per_artist=2` and `max_per_genre=3`, pushing overflow items into an overflow list that can fill remaining slots.

- Object-oriented and functional APIs:
	- `Recommender` class provides OOP interface with `recommend()` and `explain_recommendation()`.
	- `recommend_songs()` is the functional entrypoint used by `src/main.py`.

---

## 4. Retrieval / RAG Details (what the code does)

- `src/rag_recommender.py` implements a retrieval-augmented component (`RAGRecommender`) that:
	- Uses `sentence_transformers` (`all-MiniLM-L6-v2`) to produce embeddings.
	- Uses `chromadb` as an embedding store/semantic index for songs (`songs` collection) and a separate `context` collection for genre/mood documents.
	- `retrieve()` returns the most semantically similar songs for a user profile by embedding a short natural-language query built from the profile.
	- `build_index()` and `build_context_index()` create Chroma collections from the songs and knowledge-base documents respectively.
	- `generate_recommendation()` builds a prompt (including up to 3 top songs and fetched context documents) and calls Groq's chat completions API (configured with `GROQ_API_KEY`) to produce warm, human-readable explanations. The code logs to `rag_recommender.log`.

Note: the RAG pipeline requires `sentence-transformers`, `chromadb`, and a Groq/Llama API key to call the cloud LLM. If those services are not available the pipeline still supports local ranking via `recommender.py`.

---

## 5. Data

- Primary catalog: `data/songs.csv`. `load_songs()` expects these fields (CSV column names):

	`id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, popularity, release_decade, mood_tags, explicit, duration_sec, youtube_url`

- Knowledge-base docs: `data/genre_docs.json` and `data/mood_docs.json` (used by `RAGRecommender.build_context_index()`) — each entry should provide `id` and `content` fields.

- The included demo dataset is small (a classroom-sized catalog). Treat evaluation as qualitative unless extended with larger ground-truth data.

---

## 6. Strengths

- Transparent, interpretable scoring: per-song breakdown and human-readable reasons are produced by `score_song()` and `explain_recommendation()`.
- Fast deterministic baseline: the functional `recommend_songs()` and `Recommender` class allow quick experiments without heavy dependencies.
- RAG adds contextual explanations when the embedding index and Groq LLM are enabled — improving explainability and user-facing copy.

---

## 7. Limitations & Bias

- Binary genre/mood matching means non-exact or multi-genre tastes are not well-handled.
- The default scoring ignores several available song features (e.g., `valence`, `danceability`, `tempo_bpm`) unless you switch to a custom strategy that uses them — so certain musical attributes are underutilized.
- Small demo dataset can over-emphasize frequent artists/genres; diversity rerank attempts mitigation but is heuristic.
- RAG requires an external embedding model and Chroma/Groq services; using cloud LLMs can introduce cost and external-data bias.

---

## 8. Evaluation Summary (what I ran / observed)

- Unit tests in `tests/` exercise `Recommender` behavior and `score_song()` edge cases. Run them with `pytest -q`.
- Manual CLI runs (`src/main.py`) demonstrate the three ranking strategies (`balanced`, `genre_first`, `mood_first`) and show score breakdowns and ASCII tables.
- With RAG enabled (index built and Groq key set) the system returns short LLM-written explanations anchored to retrieved context documents — useful for UX but dependent on external services.

---

## 9. Future Work

- Extend scoring to incorporate `valence`, `danceability`, and `tempo_bpm` directly and expose them as strategy-weighted dimensions.
- Add configurable, persisted vector store (FAISS/Chroma with on-disk persistence) and batch embedding scripts for larger catalogs.
- Add offline fallback prompts for explanation generation when Groq/LLM calls fail, to keep UX stable.

---

## 10. Personal Reflection

Working through this repository reinforced the value of modular design: keep ranking, retrieval, and UI separate so you can iterate on one part without breaking others. The RAG pattern greatly improves conversational explanations, but introduces engineering overhead (embeddings, index management, and an LLM). For a production recommender, we'd pair the current interpretable scoring with quantitative evaluation (precision@k, nDCG) on a larger dataset.

---

See the implementation in `src/recommender.py` and `src/rag_recommender.py` for exact code paths and `tests/` for unit tests.
