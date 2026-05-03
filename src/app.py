"""Streamlit music recommender app with multi-source RAG and user profiles."""
import os
import json
import streamlit as st
from recommender import load_songs, recommend_songs
from rag_recommender import RAGRecommender, UserProfile, Song

PROFILES_PATH = "../data/profiles.json"
GENRES = ["pop", "rock", "lofi", "indie pop", "electronic", "jazz"]
MOODS = ["happy", "chill", "intense", "sad", "energetic"]
STRATEGIES = ["balanced", "genre_first", "mood_first"]
STRATEGY_LABELS = {
    "balanced": "Balanced (all factors equal)",
    "genre_first": "Genre First (prioritize genre match)",
    "mood_first": "Mood First (prioritize mood match)",
}


# ── Profile persistence ────────────────────────────────────────────────────────

def load_profiles() -> dict:
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profiles(data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(PROFILES_PATH)), exist_ok=True)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Session state defaults ─────────────────────────────────────────────────────

_defaults = {
    "genre": GENRES[0],
    "mood": MOODS[0],
    "energy": 7,
    "acousticness": 5,
    "previously_liked": "",
    "strategy": STRATEGIES[0],
    "num_recs": 5,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Cached resources ───────────────────────────────────────────────────────────

@st.cache_data
def load_song_data() -> list:
    return load_songs("../data/songs.csv")


@st.cache_data
def load_context_docs() -> list:
    docs = []
    for path in ("../data/genre_docs.json", "../data/mood_docs.json"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                docs.extend(json.load(f))
    return docs


@st.cache_resource
def build_rag(_songs_json: str, _context_json: str) -> RAGRecommender:
    rag = RAGRecommender()
    rag.build_index(json.loads(_songs_json))
    context = json.loads(_context_json)
    if context:
        rag.build_context_index(context)
    return rag


# ── Theme ──────────────────────────────────────────────────────────────────────

def set_theme():
    st.markdown("""
    <style>
    /* Page background and base text */
    .stApp { background-color: #FFFFFF; color: #000000; }

    /* Main content area */
    .main .block-container { background-color: #FFFFFF; color: #000000; }

    /* Headings */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000 !important;
    }
    h1 { text-align: center; font-size: 2.5em; margin-bottom: 20px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F8F8F8; }
    [data-testid="stSidebar"] * { color: #000000; }

    /* Primary buttons */
    .stButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }

    /* Metric labels and values */
    [data-testid="stMetricValue"] { color: #000000 !important; font-weight: bold; }
    [data-testid="stMetricLabel"] { color: #555555 !important; }

    /* Input widgets */
    .stSelectbox label, .stSlider label, .stTextInput label { color: #000000 !important; }

    /* Score badge in song cards */
    .score-badge {
        font-size: 1.4em;
        font-weight: bold;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Music Recommender", layout="wide", page_icon="🎵")
set_theme()
st.title("🎵 Music Recommender")
st.markdown(
    "<p style='text-align:center;font-size:18px;color:#555;margin-bottom:24px;'>"
    "Tell us about your music taste and get personalized recommendations.</p>",
    unsafe_allow_html=True,
)

songs = load_song_data()
context_docs = load_context_docs()

_api_key = os.environ.get("GROQ_API_KEY", "")
if not _api_key or _api_key == "your_key_here":
    st.warning(
        "⚠️ **Groq API key not set.** "
        "Open `.env` and replace `your_key_here` with your free key from "
        "[console.groq.com](https://console.groq.com), then restart the app. "
        "Recommendations still work — only the AI explanation will be unavailable."
    )

with st.spinner("Building recommendation index…"):
    rag = build_rag(json.dumps(songs), json.dumps(context_docs))

st.caption(
    f"✅ {len(songs)} songs and {len(context_docs)} knowledge-base docs "
    "indexed in ChromaDB."
)

st.divider()


# ── Sidebar: Profile management ────────────────────────────────────────────────

with st.sidebar:
    st.header("👤 Profiles")
    profiles = load_profiles()

    if profiles:
        selected_profile = st.selectbox(
            "Load a profile",
            options=["— select —"] + list(profiles.keys()),
        )
        col_load, col_delete = st.columns(2)
        with col_load:
            if (
                st.button("Load", use_container_width=True)
                and selected_profile != "— select —"
            ):
                for key in _defaults:
                    if key in profiles[selected_profile]:
                        st.session_state[key] = profiles[selected_profile][key]
                st.rerun()
        with col_delete:
            if (
                st.button("Delete", use_container_width=True)
                and selected_profile != "— select —"
            ):
                profiles.pop(selected_profile, None)
                save_profiles(profiles)
                st.success(f"Deleted '{selected_profile}'")
                st.rerun()
    else:
        st.info("No saved profiles yet.")

    st.divider()
    st.subheader("Save current settings")
    new_profile_name = st.text_input("Profile name", placeholder="e.g. Late Night Vibes")
    if st.button("Save Profile", use_container_width=True):
        if new_profile_name.strip():
            profiles[new_profile_name.strip()] = {k: st.session_state[k] for k in _defaults}
            save_profiles(profiles)
            st.success(f"Saved '{new_profile_name.strip()}'!")
        else:
            st.error("Enter a profile name first.")

    st.divider()
    with st.expander("🔬 RAG System Info"):
        st.markdown("**Indexing (Step 1)**")
        st.write(f"Songs in vector store: **{len(songs)}**")
        st.write(f"Knowledge-base docs: **{len(context_docs)}**")
        st.caption(
            "Embeddings: `all-MiniLM-L6-v2`  \n"
            "Vector DB: ChromaDB (two collections: songs + context)"
        )


# ── Main: Preferences form ─────────────────────────────────────────────────────

st.header("🎧 Your Music Preferences")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("🎼 Basic")
    genre = st.selectbox(
        "Favorite genre", GENRES,
        index=GENRES.index(st.session_state.genre) if st.session_state.genre in GENRES else 0,
        key="genre",
    )
    mood = st.selectbox(
        "Preferred mood", MOODS,
        index=MOODS.index(st.session_state.mood) if st.session_state.mood in MOODS else 0,
        key="mood",
    )

with col2:
    st.subheader("🔊 Audio")
    energy = st.slider(
        "Energy Level (1 = calm, 10 = intense)", 1, 10, key="energy",
    )
    acousticness = st.slider(
        "Acoustic Preference (1 = electronic, 10 = acoustic)", 1, 10, key="acousticness",
    )

st.subheader("💝 Personal Touch")
previously_liked = st.text_input(
    "Songs you've loved recently (optional, comma-separated)",
    placeholder="e.g. Sunrise City, Midnight Coding",
    key="previously_liked",
)

col_strat, col_num = st.columns([3, 1])
with col_strat:
    strategy = st.selectbox(
        "Recommendation Strategy", STRATEGIES,
        index=STRATEGIES.index(st.session_state.strategy)
        if st.session_state.strategy in STRATEGIES else 0,
        key="strategy", format_func=lambda x: STRATEGY_LABELS[x],
    )
with col_num:
    num_recs = st.slider("# Recommendations", 1, 10, key="num_recs")

st.markdown("")
_, col_mid, _ = st.columns([2, 1, 2])
with col_mid:
    get_recs = st.button("🎵 Get Recommendations", use_container_width=True)


# ── Recommendations ────────────────────────────────────────────────────────────

if get_recs:
    liked_list = [s.strip() for s in previously_liked.split(",") if s.strip()]
    user_prefs = {
        "favorite_genre": genre,
        "favorite_mood": mood,
        "target_energy": float(energy),
        "acousticness": float(acousticness),
        "previously_liked": liked_list,
    }

    with st.spinner("Scoring and ranking songs…"):
        recs = recommend_songs(user_prefs, songs, k=num_recs, strategy=strategy)

    if not recs:
        st.error("No recommendations found. Try adjusting your preferences.")
        st.stop()

    st.divider()
    st.header("🎉 Your Recommendations")

    user_profile = UserProfile(
        genre=genre, mood=mood,
        energy=energy / 10.0, acousticness=acousticness / 10.0,
        previously_liked=liked_list,
    )
    top_song_objects = [
        Song(
            id=s["id"], title=s["title"], artist=s["artist"],
            genre=s["genre"], mood=s["mood"],
            energy=s["energy"], acousticness=s["acousticness"],
            mood_tags=s.get("mood_tags", ""),
        )
        for s, _, _ in recs[:3]
    ]

    # ── RAG pipeline (visible steps) ──────────────────────────────────────────
    with st.expander("🔬 RAG Pipeline — How this was generated", expanded=False):
        st.markdown("**Step 2 — Semantic Retrieval (ChromaDB)**")
        semantic_hits = rag.retrieve(user_profile, k=5)
        if semantic_hits:
            for hit in semantic_hits:
                st.caption(
                    f"• *{hit['title']}* by {hit['artist']} "
                    f"({hit['genre']} / {hit['mood']})"
                )
        else:
            st.caption("No semantic hits — index may not be built.")

        st.markdown("**Step 3 — Multi-Source Knowledge Retrieval**")
        context_preview = rag.retrieve_context(user_profile, k=2)
        if context_preview:
            for i, doc in enumerate(context_preview, 1):
                st.info(f"**Context doc {i}:** {doc[:220]}…")
        else:
            st.caption("No context docs found.")

        st.markdown("**Step 4 — LLM Generation via Groq (`llama-3.3-70b-versatile`)**")
        st.caption(
            "The top songs + retrieved context docs are assembled into a prompt "
            "and sent to the Groq LLM to produce a personalised explanation."
        )

    # ── AI explanation ─────────────────────────────────────────────────────────
    with st.spinner("Generating AI explanation…"):
        try:
            overall_explanation = rag.generate_recommendation(user_profile, top_song_objects)
            st.markdown("### 💭 AI Recommendation")
            st.info(overall_explanation)
        except (ValueError, RuntimeError, OSError) as e:
            st.warning(f"AI explanation unavailable: {e}")

    # ── Song cards ─────────────────────────────────────────────────────────────
    st.markdown("### 🎵 Song Details")

    for i, (song, score, reasons) in enumerate(recs, 1):
        with st.container(border=True):
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(
                    f"<div class='score-badge'>#{i}</div>"
                    f"<div style='font-size:1.1em;'>{score:.1f}<span style='font-size:0.7em;color:#888;'>/10</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(f"**{song['title']}** by *{song['artist']}*")
                st.caption(f"Genre: {song['genre']}  •  Mood: {song['mood']}  •  Energy: {song['energy']:.2f}  •  Acousticness: {song['acousticness']:.2f}")
                youtube_url = song.get("youtube_url", "")
                if youtube_url:
                    st.markdown(
                        f'<a href="{youtube_url}" target="_blank" style="'
                        "display:inline-block;padding:3px 12px;"
                        "background:#000;color:#fff;"
                        "border-radius:4px;font-size:13px;text-decoration:none;\">"
                        "▶ Search on YouTube</a>",
                        unsafe_allow_html=True,
                    )
            with st.expander("🔍 Why this song?"):
                for line in reasons.splitlines():
                    if line.strip():
                        st.markdown(f"- {line.strip()}")
