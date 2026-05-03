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
    """Load saved user profiles from disk."""
    if os.path.exists(PROFILES_PATH):
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profiles(data: dict) -> None:
    """Persist user profiles to disk."""
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
    """Load song catalogue from CSV."""
    return load_songs("../data/songs.csv")


@st.cache_data
def load_context_docs() -> list:
    """Load and merge genre + mood knowledge-base documents."""
    docs = []
    for path in ("../data/genre_docs.json", "../data/mood_docs.json"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                docs.extend(json.load(f))
    return docs


@st.cache_resource
def build_rag(_songs_json: str, _context_json: str) -> RAGRecommender:
    """Build song and context ChromaDB indexes once per session."""
    rag = RAGRecommender()
    rag.build_index(json.loads(_songs_json))
    context = json.loads(_context_json)
    if context:
        rag.build_context_index(context)
    return rag


# ── Theme ──────────────────────────────────────────────────────────────────────

def set_theme():
    """Inject monochrome CSS theme."""
    st.markdown("""
    <style>
    * { background-color: #FFFFFF !important; color: #000000 !important; }
    h1 { text-align: center !important; font-size: 2.5em !important;
         margin-bottom: 30px !important; }
    .stButton>button {
        background-color: #FFFFFF !important; color: #000000 !important;
        border: 2px solid #000000 !important; border-radius: 8px !important;
        font-weight: bold !important; font-size: 18px !important;
        padding: 12px 24px !important; transition: all 0.3s ease !important;
        min-height: 44px !important;
    }
    .stButton>button:hover {
        background-color: #000000 !important; color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Music Recommender", page_icon="🎵")
set_theme()
st.title("🎵 Music Recommender")
st.markdown(
    "<div style='text-align:center;margin-bottom:30px;'>"
    "<h2>Discover Your Perfect Playlist</h2>"
    "<p style='font-size:18px;'>Tell us about your music taste and get "
    "personalized recommendations!</p></div>",
    unsafe_allow_html=True,
)

songs = load_song_data()
context_docs = load_context_docs()
st.success(
    f"✅ Loaded {len(songs)} songs "
    f"and {len(context_docs)} knowledge-base documents."
)

_api_key = os.environ.get("GROQ_API_KEY", "")
if not _api_key or _api_key == "your_key_here":
    st.warning(
        "⚠️ **Groq API key not set.** "
        "Open `.env` in the project root and replace `your_key_here` with your "
        "free key from [console.groq.com](https://console.groq.com), "
        "then restart the app. Recommendations still work — only the AI "
        "explanation will be unavailable."
    )

with st.spinner("Building recommendation index..."):
    rag = build_rag(json.dumps(songs), json.dumps(context_docs))


# ── Sidebar: Profile management ────────────────────────────────────────────────

with st.sidebar:
    st.header("👤 Your Profiles")
    profiles = load_profiles()

    if profiles:
        selected_profile = st.selectbox(
            "Load a profile",
            options=["-- select --"] + list(profiles.keys()),
        )
        col_load, col_delete = st.columns(2)
        with col_load:
            if (
                st.button("Load", use_container_width=True)
                and selected_profile != "-- select --"
            ):
                for key in _defaults:
                    if key in profiles[selected_profile]:
                        st.session_state[key] = profiles[selected_profile][key]
                st.rerun()
        with col_delete:
            if (
                st.button("Delete", use_container_width=True)
                and selected_profile != "-- select --"
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


# ── Main: Preferences form ─────────────────────────────────────────────────────

st.header("🎧 Your Music Preferences")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎼 Basic Preferences")
    genre = st.selectbox(
        "What's your favorite genre?", GENRES,
        index=GENRES.index(st.session_state.genre) if st.session_state.genre in GENRES else 0,
        key="genre", help="Choose the music genre you enjoy most",
    )
    mood = st.selectbox(
        "What's your preferred mood?", MOODS,
        index=MOODS.index(st.session_state.mood) if st.session_state.mood in MOODS else 0,
        key="mood", help="Select the emotional vibe you're looking for",
    )

with col2:
    st.subheader("🔊 Audio Preferences")
    energy = st.slider(
        "Energy Level (1-10)", 1, 10, key="energy",
        help="1=very calm, 10=very energetic",
    )
    acousticness = st.slider(
        "Acoustic Preference (1-10)", 1, 10, key="acousticness",
        help="1=electronic/heavy production, 10=acoustic/natural",
    )

st.subheader("💝 Personal Touch")
previously_liked = st.text_input(
    "Songs you've loved recently (optional)",
    placeholder="e.g., Sunrise City, Midnight Coding",
    key="previously_liked",
    help="Enter song titles separated by commas",
)

st.subheader("🎯 Recommendation Settings")
strategy = st.selectbox(
    "Recommendation Strategy", STRATEGIES,
    index=STRATEGIES.index(st.session_state.strategy)
    if st.session_state.strategy in STRATEGIES else 0,
    key="strategy", format_func=lambda x: STRATEGY_LABELS[x],
)
num_recs = st.slider("Number of Recommendations", 1, 10, key="num_recs")

_, col_mid, _ = st.columns([1, 1, 1])
with col_mid:
    get_recs = st.button("🎵 Get My Recommendations!", use_container_width=True)


# ── Recommendations ────────────────────────────────────────────────────────────

if get_recs:
    with st.spinner("🎵 Finding your perfect songs..."):
        liked_list = [s.strip() for s in previously_liked.split(",") if s.strip()]
        user_prefs = {
            "favorite_genre": genre,
            "favorite_mood": mood,
            "target_energy": float(energy),
            "acousticness": float(acousticness),
            "previously_liked": liked_list,
        }
        recs = recommend_songs(user_prefs, songs, k=num_recs, strategy=strategy)

    if recs:
        st.header("🎉 Your Personalized Recommendations")

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

        with st.spinner("Generating explanation with Groq..."):
            try:
                overall_explanation = rag.generate_recommendation(
                    user_profile, top_song_objects
                )
                st.markdown("### 💭 Overall Recommendation")
                st.info(overall_explanation)
            except (ValueError, RuntimeError, OSError) as e:
                st.warning(f"Could not generate AI explanation: {e}")

        st.markdown("### 🎵 Detailed Recommendations")

        for i, (song, score, reasons) in enumerate(recs, 1):
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.metric(f"#{i}", f"{score:.1f}/10 ⭐")
                with c2:
                    st.markdown(f"**{song['title']}** by *{song['artist']}*")
                    st.caption(f"Genre: {song['genre']} • Mood: {song['mood']}")
                    youtube_url = song.get("youtube_url", "")
                    if youtube_url:
                        st.markdown(
                            f'<a href="{youtube_url}" target="_blank" style="'
                            'display:inline-block;padding:4px 12px;'
                            'background:#000000;color:#ffffff;'
                            'border:1px solid #000000;border-radius:4px;'
                            'font-size:13px;text-decoration:none;">'
                            "▶ Search on YouTube</a>",
                            unsafe_allow_html=True,
                        )
                with st.expander("🔍 Why this song?"):
                    st.write(reasons)
    else:
        st.error("❌ No recommendations found. Try adjusting your preferences!")
