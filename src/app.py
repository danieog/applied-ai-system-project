import streamlit as st
from recommender import load_songs, recommend_songs
from rag_recommender import RAGRecommender, UserProfile, Song

st.set_page_config(page_title="Music Recommender", page_icon="🎵")

st.title("🎵 Music Recommender")

# Load songs
@st.cache_data
def load_data():
    songs = load_songs("../data/songs.csv")
    return songs

songs = load_data()
st.success(f"Loaded {len(songs)} songs from the database.")

# User input
st.header("Your Music Preferences")

col1, col2 = st.columns(2)

with col1:
    genre = st.selectbox("Favorite Genre", ["pop", "rock", "lofi", "indie pop", "electronic", "jazz"])
    mood = st.selectbox("Favorite Mood", ["happy", "chill", "intense", "sad", "energetic"])

with col2:
    energy = st.slider("Energy Level (1-10)", 1, 10, 7)
    acousticness = st.slider("Acousticness Preference (1-10)", 1, 10, 5)

previously_liked = st.text_input("Previously Liked Songs (comma-separated titles)", "")

strategy = st.selectbox("Recommendation Strategy", ["balanced", "genre_first", "mood_first"])

num_recs = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    # Prepare user profile
    user_prefs = {
        "favorite_genre": genre,
        "favorite_mood": mood,
        "target_energy": energy / 10.0,  # Convert to 0-1 scale
        "acousticness": acousticness / 10.0,
        "previously_liked": [s.strip() for s in previously_liked.split(",") if s.strip()],
    }

    # Get recommendations
    recs = recommend_songs(user_prefs, songs, k=num_recs, strategy=strategy)

    if recs:
        st.header("Your Recommendations")

        # Create RAG recommender for explanations
        user_profile = UserProfile(
            genre=genre,
            mood=mood,
            energy=energy / 10.0,
            acousticness=acousticness / 10.0,
            previously_liked=user_prefs["previously_liked"]
        )
        rag = RAGRecommender()

        # Convert top songs to Song objects for RAG
        top_song_objects = [Song(**song) for song, _, _ in recs[:3]]

        # Get overall explanation for top 3
        overall_explanation = rag.generate_recommendation(user_profile, top_song_objects)

        st.write("**Overall Recommendation:**")
        st.write(overall_explanation)

        st.header("Detailed Recommendations")

        for i, (song, score, reasons) in enumerate(recs, 1):
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(f"#{i}", f"{score:.1f}/10")
                with col2:
                    st.subheader(f"{song['title']} by {song['artist']}")
                    st.write(f"Genre: {song['genre']} | Mood: {song['mood']}")

                with st.expander("Score Breakdown"):
                    st.write("\n".join(reasons))
    else:
        st.error("No recommendations found. Try adjusting your preferences.")