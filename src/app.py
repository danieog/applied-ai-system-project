import streamlit as st
from recommender import load_songs, recommend_songs
from rag_recommender import RAGRecommender, UserProfile, Song

# Custom CSS for theme - Accessibility focused
def set_theme():
    st.markdown("""
    <style>
    /* Overall background - cream */
    .stApp {
        background-color: #FAF0E6 !important; /* Cream/linen background */
        color: #000000 !important;
        font-size: 16px !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #F0F0F0 !important; /* Light gray */
        color: #000000 !important;
    }
    
    /* All text elements - high contrast */
    p, div, span, label {
        color: #000000 !important;
        font-size: 16px !important;
    }
    
    /* Headers - large and high contrast */
    h1, h2, h3 {
        color: #000000 !important; /* Black for maximum contrast */
        font-family: 'Arial', sans-serif !important;
        font-weight: bold !important;
    }
    
    /* Title */
    h1 {
        text-align: center !important;
        font-size: 2.5em !important; /* Larger for accessibility */
        margin-bottom: 30px !important;
    }
    
    /* Buttons - high contrast with clear focus */
    .stButton>button {
        background-color: #0066CC !important; /* High contrast blue */
        color: #FFFFFF !important; /* White text */
        border: 3px solid #004499 !important; /* Thicker border */
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 18px !important; /* Larger text */
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
        min-height: 44px !important; /* Minimum touch target size */
    }
    
    .stButton>button:hover {
        background-color: #004499 !important;
        transform: scale(1.02) !important;
    }
    
    .stButton>button:focus {
        outline: 4px solid #FF6B35 !important; /* Orange focus ring */
        outline-offset: 2px !important;
    }
    
    /* Sliders - high contrast */
    .stSlider .st-bq {
        background-color: #0066CC !important; /* Blue track */
    }
    
    .stSlider [role="slider"] {
        background-color: #0066CC !important;
        border: 2px solid #004499 !important;
    }
    
    /* Select boxes - match cream background */
    .stSelectbox div[data-baseweb="select"],
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] > div:first-child {
        background-color: #FAF0E6 !important;
        border: 3px solid #000000 !important;
        border-radius: 8px !important;
        color: #000000 !important;
        font-size: 16px !important;
        min-height: 44px !important;
    }

    .stSelectbox div[data-baseweb="select"]:focus {
        outline: 4px solid #FF6B35 !important;
        outline-offset: 2px !important;
    }

    /* Dropdown menu options - match cream background */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"] {
        background-color: #FAF0E6 !important;
        border: 2px solid #000000 !important;
    }

    div[data-baseweb="popover"] div[role="option"],
    ul[data-baseweb="menu"] li {
        background-color: #FAF0E6 !important;
        color: #000000 !important;
        font-size: 16px !important;
        padding: 12px !important;
        min-height: 44px !important;
    }

    div[data-baseweb="popover"] div[role="option"]:hover,
    ul[data-baseweb="menu"] li:hover {
        background-color: #EEE0D0 !important;
    }

    div[data-baseweb="popover"] div[role="option"][aria-selected="true"],
    ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #E8D8C4 !important;
    }
    
    /* Text inputs - match cream background */
    .stTextInput input {
        background-color: #FAF0E6 !important;
        border: 3px solid #000000 !important;
        border-radius: 8px !important;
        color: #000000 !important;
        font-size: 16px !important;
        padding: 12px !important;
        min-height: 44px !important;
    }
    
    .stTextInput input:focus {
        outline: 4px solid #FF6B35 !important;
        outline-offset: 2px !important;
    }
    
    /* Metrics - high contrast */
    .stMetric {
        background-color: #F0F0F0 !important; /* Light gray */
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        color: #000000 !important;
    }
    
    .stMetric label {
        font-size: 18px !important;
        font-weight: bold !important;
    }
    
    .stMetric .metric-value {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    /* Containers - match cream background */
    .stContainer {
        background-color: #FAF0E6 !important;
        border: 3px solid #000000 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        margin: 20px 0 !important;
    }
    
    /* Expanders - high contrast */
    .streamlit-expanderHeader {
        background-color: #F0F0F0 !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 12px !important;
        min-height: 44px !important;
    }
    
    .streamlit-expanderHeader:focus {
        outline: 4px solid #FF6B35 !important;
        outline-offset: 2px !important;
    }
    
    /* Success messages - high contrast */
    .stSuccess {
        background-color: #D4EDDA !important; /* Light green */
        border: 2px solid #155724 !important; /* Dark green border */
        border-radius: 8px !important;
        color: #155724 !important;
        font-size: 16px !important;
    }
    
    /* Error messages - high contrast */
    .stError {
        background-color: #F8D7DA !important; /* Light red */
        border: 2px solid #721C24 !important; /* Dark red border */
        border-radius: 8px !important;
        color: #721C24 !important;
        font-size: 16px !important;
    }
    
    /* Info messages - high contrast */
    .stInfo {
        background-color: #D1ECF1 !important; /* Light blue */
        border: 2px solid #0C5460 !important; /* Dark blue border */
        border-radius: 8px !important;
        color: #0C5460 !important;
        font-size: 16px !important;
    }
    
    /* Loading spinner - ensure visibility */
    .stSpinner {
        font-size: 16px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="Music Recommender", page_icon="🎵")

set_theme()

st.title("🎵 Music Recommender")

st.markdown("""
<div style='text-align: center; margin-bottom: 30px;'>
    <h2 style='color: #8B4513;'>Discover Your Perfect Playlist</h2>
    <p style='font-size: 18px; color: #666;'>Tell us about your music taste and get personalized recommendations!</p>
</div>
""", unsafe_allow_html=True)

# Load songs
@st.cache_data
def load_data():
    songs = load_songs("../data/songs.csv")
    return songs

songs = load_data()
st.success(f"✅ Loaded {len(songs)} songs from our database!")

# User input
st.header("🎧 Your Music Preferences")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎼 Basic Preferences")
    genre = st.selectbox("What's your favorite genre?", 
                        ["pop", "rock", "lofi", "indie pop", "electronic", "jazz"],
                        help="Choose the music genre you enjoy most")
    
    mood = st.selectbox("What's your preferred mood?", 
                       ["happy", "chill", "intense", "sad", "energetic"],
                       help="Select the emotional vibe you're looking for")

with col2:
    st.subheader("🔊 Audio Preferences")
    energy = st.slider("Energy Level (1-10)", 1, 10, 7, 
                      help="How energetic do you want your music? 1=very calm, 10=very energetic")
    
    acousticness = st.slider("Acoustic Preference (1-10)", 1, 10, 5,
                            help="How acoustic do you prefer? 1=electronic/heavy production, 10=acoustic/natural")

st.subheader("💝 Personal Touch")
previously_liked = st.text_input("Songs you've loved recently (optional)", 
                                placeholder="e.g., Sunrise City, Midnight Coding",
                                help="Enter song titles separated by commas to get more personalized recommendations")

st.subheader("🎯 Recommendation Settings")
strategy = st.selectbox("Recommendation Strategy", 
                       ["balanced", "genre_first", "mood_first"],
                       format_func=lambda x: {
                           "balanced": "Balanced (all factors equal)",
                           "genre_first": "Genre First (prioritize genre match)",
                           "mood_first": "Mood First (prioritize mood match)"
                       }[x])

num_recs = st.slider("Number of Recommendations", 1, 10, 5,
                    help="How many song recommendations would you like?")

# Center the button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    get_recs = st.button("🎵 Get My Recommendations!", use_container_width=True)

if get_recs:
    with st.spinner("🎵 Finding your perfect songs..."):
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
        st.header("🎉 Your Personalized Recommendations")

        # Create RAG recommender for explanations
        user_profile = UserProfile(
            genre=genre,
            mood=mood,
            energy=energy / 10.0,
            acousticness=acousticness / 10.0,
            previously_liked=user_prefs["previously_liked"]
        )
        rag = RAGRecommender()

        # Convert top songs to Song objects for RAG (only pass required fields)
        top_song_objects = [
            Song(
                id=song['id'],
                title=song['title'],
                artist=song['artist'],
                genre=song['genre'],
                mood=song['mood'],
                energy=song['energy'],
                acousticness=song['acousticness']
            ) for song, _, _ in recs[:3]
        ]

        # Get overall explanation for top 3
        overall_explanation = rag.generate_recommendation(user_profile, top_song_objects)

        st.markdown("### 💭 Overall Recommendation")
        st.info(overall_explanation)

        st.markdown("### 🎵 Detailed Recommendations")

        for i, (song, score, reasons) in enumerate(recs, 1):
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(f"#{i}", f"{score:.1f}/10 ⭐")
                with col2:
                    st.markdown(f"**{song['title']}** by *{song['artist']}*")
                    st.caption(f"Genre: {song['genre']} • Mood: {song['mood']}")

                with st.expander("🔍 Why this song?"):
                    st.write(reasons)
    else:
        st.error("❌ No recommendations found. Try adjusting your preferences!")