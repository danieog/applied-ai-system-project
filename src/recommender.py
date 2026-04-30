from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Ranking strategies
# Each strategy is a weight dict whose values must sum to 1.0.
# Keys map directly to the seven scored dimensions in score_song().
# ---------------------------------------------------------------------------
RANKING_STRATEGIES: Dict[str, Dict[str, float]] = {
    # Balanced — equal emphasis on genre and mood
    "balanced": {
        "genre":        0.25,
        "mood":         0.15,
        "energy":       0.20,
        "acousticness": 0.10,
        "popularity":   0.10,
        "decade":       0.10,
        "mood_tags":    0.10,
    },
    # Genre first — genre match dominates; mood is a tiebreaker
    "genre_first": {
        "genre":        0.45,
        "mood":         0.08,
        "energy":       0.15,
        "acousticness": 0.08,
        "popularity":   0.08,
        "decade":       0.08,
        "mood_tags":    0.08,
    },
    # Mood first — mood match dominates; genre is a tiebreaker
    "mood_first": {
        "genre":        0.08,
        "mood":         0.40,
        "energy":       0.15,
        "acousticness": 0.08,
        "popularity":   0.08,
        "decade":       0.08,
        "mood_tags":    0.13,
    },
}

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # New features (default values preserve backward compatibility with existing tests)
    popularity: int = 50            # 0–100 chart/stream popularity
    release_decade: int = 2010      # e.g. 1990, 2000, 2010, 2020
    mood_tags: str = ""             # pipe-separated detail tags, e.g. "upbeat|summer|driving"
    explicit: bool = False          # True if song contains explicit content
    duration_sec: int = 210         # song length in seconds

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    """
    genre: str
    mood: str
    energy: float  # 1-10 scale
    acousticness: float  # 1-10 scale
    previously_liked: List[str] = field(default_factory=list)

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _song_to_dict(self, song: Song) -> Dict:
        return {
            "id": song.id,
            "title": song.title,
            "artist": song.artist,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
            "popularity": song.popularity,
            "release_decade": song.release_decade,
            "mood_tags": song.mood_tags,
            "explicit": int(song.explicit),
            "duration_sec": song.duration_sec,
        }

    def _profile_to_dict(self, user: UserProfile) -> Dict:
        return {
            "genre": user.genre,
            "mood": user.mood,
            "energy": user.energy,
            "acousticness": user.acousticness,
            "previously_liked": user.previously_liked,
        }

    def _rank(
        self,
        user: UserProfile,
        k: int,
    ) -> List[Song]:
        user_dict  = self._profile_to_dict(user)
        id_to_song = {s.id: s for s in self.songs}
        scored = [
            (self._song_to_dict(s), *score_song(user_dict, self._song_to_dict(s)))
            for s in self.songs
        ]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        diverse = diversity_rerank(ranked, k)
        return [id_to_song[sd["id"]] for sd, _, _ in diverse]

    def recommend(
        self,
        user: UserProfile,
        k: int = 5,
    ) -> List[Song]:
        return self._rank(user, k)

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_dict = self._profile_to_dict(user)
        _, reasons = score_song(user_dict, self._song_to_dict(song))
        return "\n".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    import csv
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_dict = {
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
                # New features
                "popularity": int(row.get("popularity", 50)),
                "release_decade": int(row.get("release_decade", 2010)),
                "mood_tags": row.get("mood_tags", ""),
                "explicit": int(row.get("explicit", 0)),
                "duration_sec": int(row.get("duration_sec", 210)),
            }
            songs.append(song_dict)
    return songs

def score_song(
    user_prefs: Dict,
    song: Dict,
    weights: Dict[str, float] = None,
) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences using provided weights.
    """
    if weights is None:
        weights = {
            "genre": 0.3,
            "mood": 0.25,
            "energy": 0.25,
            "acousticness": 0.15,
            "feedback": 0.05,
        }
    
    reasons = []

    # Normalize song features to 1-10 scale (CSV has 0-1)
    song_energy = song['energy'] * 10
    song_acousticness = song['acousticness'] * 10

    # Genre score
    genre_score = 10 if song['genre'] == user_prefs.get('favorite_genre', user_prefs.get('genre', '')) else 0
    reasons.append(f"Genre: {'match' if genre_score == 10 else 'no match'}")

    # Mood score
    mood_score = 10 if song['mood'] == user_prefs.get('favorite_mood', user_prefs.get('mood', '')) else 0
    reasons.append(f"Mood: {'match' if mood_score == 10 else 'no match'}")

    # Energy score - distance based
    energy_diff = abs(user_prefs.get('target_energy', user_prefs.get('energy', 5)) - song_energy)
    energy_score = max(0, 10 - energy_diff)
    reasons.append(f"Energy: {song_energy:.1f} vs {user_prefs.get('target_energy', user_prefs.get('energy', 5)):.1f} (diff={energy_diff:.1f})")

    # Acousticness score
    acoustic_diff = abs(user_prefs.get('acousticness', 5) - song_acousticness)
    acoustic_score = max(0, 10 - acoustic_diff)
    reasons.append(f"Acousticness: {song_acousticness:.1f} vs {user_prefs.get('acousticness', 5):.1f} (diff={acoustic_diff:.1f})")

    # Feedback score
    feedback_score = 0
    if song['title'] in user_prefs.get('previously_liked', []):
        feedback_score = 10
        reasons.append("Liked before")
    elif song.get('id') in user_prefs.get('likes', []):
        feedback_score = 10
        reasons.append("You liked this song before")
    elif song.get('id') in user_prefs.get('skips', []):
        feedback_score = -5  # penalty
        reasons.append("You skipped this song before")

    # Weighted total score
    total_score = (
        genre_score * weights.get('genre', 0) +
        mood_score * weights.get('mood', 0) +
        energy_score * weights.get('energy', 0) +
        acoustic_score * weights.get('acousticness', 0) +
        max(0, feedback_score) * weights.get('feedback', 0)
    )

    # Apply feedback penalty if negative
    if feedback_score < 0:
        total_score += feedback_score

    return max(0, min(10, total_score)), reasons

def diversity_rerank(
    ranked: List[Tuple[Dict, float, List[str]]],
    k: int,
    max_per_artist: int = 2,
    max_per_genre: int = 3,
) -> List[Tuple[Dict, float, List[str]]]:
    """
    Greedy diversity pass over a pre-sorted scored list.

    Iterates through songs in score order and greedily selects each one only
    if it doesn't exceed max_per_artist or max_per_genre in the results so far.
    Songs that are pushed down have a note appended to their reasons and are
    placed after the diversity-selected set if slots still remain.

    Parameters
    ----------
    ranked          : scored list — each item is (song_dict, score, reasons_list)
    k               : number of results to return
    max_per_artist  : max songs from the same artist allowed in the top-k
    max_per_genre   : max songs from the same genre allowed in the top-k
    """
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}
    selected: List[Tuple[Dict, float, List[str]]] = []
    overflow: List[Tuple[Dict, float, List[str]]] = []

    for song, score, reasons in ranked:
        artist = song["artist"]
        genre  = song["genre"]
        a_count = artist_counts.get(artist, 0)
        g_count = genre_counts.get(genre, 0)

        pushed_reasons: List[str] = []
        if a_count >= max_per_artist:
            pushed_reasons.append(
                f"artist '{artist}' already has {a_count} song(s) in top results"
            )
        if g_count >= max_per_genre:
            pushed_reasons.append(
                f"genre '{genre}' already has {g_count} song(s) in top results"
            )

        if pushed_reasons:
            overflow.append((
                song, score,
                reasons + [f"[Diversity: pushed down — {'; '.join(pushed_reasons)}]"],
            ))
        else:
            artist_counts[artist] = a_count + 1
            genre_counts[genre]   = g_count + 1
            selected.append((song, score, reasons))
            if len(selected) == k:
                break

    # Fill any remaining slots with the best overflow songs
    for item in overflow:
        if len(selected) >= k:
            break
        selected.append(item)

    return selected


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    strategy: str = "balanced",
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    weights = RANKING_STRATEGIES.get(strategy, RANKING_STRATEGIES["balanced"])
    scored = [(song, *score_song(user_prefs, song, weights)) for song in songs]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    top_k = ranked[:k]
    return [(song, score, "\n".join(reasons)) for song, score, reasons in top_k]
