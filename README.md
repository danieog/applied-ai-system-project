# MelodAI 

MelodAI is a lightweight recommendation system built during Module 3 of the Applied AI Systems curriculum. Its original goal was to explore hybrid recommendation techniques (content + retrieval-augmented generation) and expose them through both a command-line and a simple web interface so non-technical users can get personalized music suggestions and short natural-language explanations.

**Title & Summary**

MelodAI delivers personalized song suggestions based on user preferences (genre, mood, keywords). It combines a classical recommender core with a retrieval-augmented component that uses short document embeddings and lookup to provide contextualized suggestions and rationale. This project matters because it demonstrates end-to-end AI system design: data, model logic, UI, and testing — all reproducible and easy to extend.

**Architecture Overview**

- **Data**: `data/songs.csv`, `data/genre_docs.json`, and `data/mood_docs.json` store the catalog and short content used for retrieval.
- **Core logic**: `src/recommender.py` implements the baseline recommendation logic (filters, scoring), while `src/rag_recommender.py` adds retrieval-augmented rationale generation.
- **Interfaces**: `src/main.py` provides a CLI entrypoint; `src/app.py` runs a Streamlit web UI for interactive use.
- **Tests**: `tests/` contains unit tests for the recommender and RAG components.

The system is intentionally modular so each component (data, ranker, retriever, UI) can be replaced or scaled independently.

**Setup Instructions**

1. Create a Python environment (recommended):

   ```bash
   python -m venv .venv
   # On Windows
   .\.venv\Scripts\activate
   # On macOS / Linux
   # source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the command-line demo:

   ```bash
   python src/main.py
   ```

4. Run the web app (Streamlit):

   ```bash
   streamlit run src/app.py
   ```

5. Run unit tests:

   ```bash
   pytest -q
   ```

**Sample Interactions**

Below are example inputs and the type of outputs the system returns (these examples illustrate expected behavior; actual text may vary slightly):

- Example 1 — relaxed indie suggestions

  Input (CLI or UI form):

  - Genre: `indie`
  - Mood: `chill`

  Output (top 3 results with rationale):

  - 1) "Lazy River" — matched genre `indie`, tempo and instrumentation align with `chill` mood. Confidence: 0.87
  - 2) "Evening Streets" — retrieved doc describing mellow guitar textures; fits user's requested vibe. Confidence: 0.81
  - 3) "Paper Boats" — content-similarity to other liked indie ballads; good for low-energy playlists. Confidence: 0.78

- Example 2 — high-energy pop workout list

  Input:

  - Genre: `pop`
  - Mood: `energetic`

  Output:

  - 1) "Run the Lights" — high-tempo pop track, strong beat for workouts. Confidence: 0.92
  - 2) "Electric Feelings" — retrieved passages highlight driving rhythm and hook. Confidence: 0.88

- Example 3 — keyword-driven contextual suggestion (RAG use)

  Input:

  - Keywords: `summer road trip, sunrise`

  Output:

  - 1) "Sunrise Drive" — retrieval found a document describing songs used in road-trip playlists; recommendation matches the imagery. Rationale: "warm acoustic opening, steady tempo fits sunrise driving." Confidence: 0.83

These examples show the dual behavior: (1) filtering/scoring for direct preference matches, and (2) retrieval-backed rationale when the user provides richer context.

**Design Decisions & Trade-offs**

- Chose a hybrid approach (filter + RAG) to balance accuracy and explainability: the core ranker is fast and deterministic; RAG provides natural-language context but adds complexity and potential latency.
- Data simplicity: the project uses CSV/JSON for portability and easy inspection. Trade-off: not ideal for very large catalogs where a database or vector DB would scale better.
- UI choice: Streamlit was selected for rapid prototyping and accessibility. Trade-off: limited control over production-grade UX and concurrency.
- Model footprint: kept model usage lightweight to avoid large dependencies; the RAG component uses small, local retrieval steps rather than large remote LLM calls to make the demo reproducible offline.

**Testing Summary**

- Unit tests for core ranking logic and retrieval helpers are located in `tests/` and are intended to cover typical edge cases (no matches, multiple scores, tie-breaking).
- Manual validation: the Streamlit app is the easiest way to exercise end-to-end behavior and confirm that UI inputs map to the recommender pipeline.
- Known limitations: the demo dataset is small and synthetic in places — expect noisier results on very sparse inputs. For production, add larger training/validation datasets and more robust evaluation metrics (precision@k, nDCG).

**Reflection**

This project taught me how to design an end-to-end AI-powered feature with pragmatic constraints: keep components decoupled, make reproducible setups, and prefer simple solutions that are easy to reason about. I learned the practical trade-offs between explainability and latency, and how retrieval (RAG) can improve user-facing explanations even with modest resources. One thing I would add in the future is having favorite artists and an account sign-in/sign-up feature where users can create their work.

