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
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # New preference fields (all optional with sensible defaults)
    preferred_decade: Optional[int] = None   # e.g. 2020; None = no era preference
    preferred_mood_tags: List[str] = field(default_factory=list)  # e.g. ["upbeat","summer"]
    allow_explicit: bool = True              # set False to penalise explicit tracks
    min_popularity: int = 0                  # filter floor; songs below this are penalised

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
            "favorite_genre": user.favorite_genre,
            "favorite_mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "preferred_decade": user.preferred_decade,
            "preferred_mood_tags": user.preferred_mood_tags,
            "allow_explicit": user.allow_explicit,
            "min_popularity": user.min_popularity,
        }

    def _rank(
        self,
        user: UserProfile,
        k: int,
        strategy: str,
        max_per_artist: int = 2,
        max_per_genre: int = 3,
    ) -> List[Song]:
        user_dict  = self._profile_to_dict(user)
        id_to_song = {s.id: s for s in self.songs}
        scored = [
            (self._song_to_dict(s), *score_song(user_dict, self._song_to_dict(s), strategy))
            for s in self.songs
        ]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        diverse = diversity_rerank(ranked, k, max_per_artist, max_per_genre)
        return [id_to_song[sd["id"]] for sd, _, _ in diverse]

    def recommend(
        self,
        user: UserProfile,
        k: int = 5,
        max_per_artist: int = 2,
        max_per_genre: int = 3,
    ) -> List[Song]:
        """Balanced strategy (default)."""
        return self._rank(user, k, "balanced", max_per_artist, max_per_genre)

    def recommend_genre_first(
        self,
        user: UserProfile,
        k: int = 5,
        max_per_artist: int = 2,
        max_per_genre: int = 3,
    ) -> List[Song]:
        """Genre-first strategy: genre match carries 45 % of the score."""
        return self._rank(user, k, "genre_first", max_per_artist, max_per_genre)

    def recommend_mood_first(
        self,
        user: UserProfile,
        k: int = 5,
        max_per_artist: int = 2,
        max_per_genre: int = 3,
    ) -> List[Song]:
        """Mood-first strategy: mood match carries 40 % of the score."""
        return self._rank(user, k, "mood_first", max_per_artist, max_per_genre)

    def explain_recommendation(self, user: UserProfile, song: Song, strategy: str = "balanced") -> str:
        user_dict = self._profile_to_dict(user)
        _, reasons = score_song(user_dict, self._song_to_dict(song), strategy)
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
    strategy: str = "balanced",
) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    strategy — one of "balanced" | "genre_first" | "mood_first"
      balanced   : Genre 25 % / Mood 15 % / Energy 20 % / …
      genre_first: Genre 45 % / Mood  8 % / Energy 15 % / …
      mood_first : Genre  8 % / Mood 40 % / Energy 15 % / …
    Plus feedback adjustment: liked +2, skipped -2 (capped to [0, 10]).
    Explicit penalty: -3 if user disallows explicit content.
    """
    weights = RANKING_STRATEGIES.get(strategy, RANKING_STRATEGIES["balanced"])
    reasons = [f"[Strategy: {strategy}]"]

    # --- Genre score (25%) — categorical exact match ---
    if song['genre'] == user_prefs['favorite_genre']:
        genre_score = 10
        reasons.append(f"Genre matches your favorite '{user_prefs['favorite_genre']}' (+10)")
    else:
        genre_score = 0
        reasons.append(f"Genre '{song['genre']}' doesn't match '{user_prefs['favorite_genre']}' (+0)")

    # --- Mood score (15%) — categorical exact match ---
    if song['mood'] == user_prefs['favorite_mood']:
        mood_score = 10
        reasons.append(f"Mood matches your favorite '{user_prefs['favorite_mood']}' (+10)")
    else:
        mood_score = 0
        reasons.append(f"Mood '{song['mood']}' doesn't match '{user_prefs['favorite_mood']}' (+0)")

    # --- Energy score (20%) — distance-based ---
    energy_diff = abs(user_prefs['target_energy'] - song['energy'])
    energy_score = 10 - (10 * energy_diff)
    reasons.append(
        f"Energy {song['energy']:.2f} vs target {user_prefs['target_energy']:.2f} "
        f"(diff={energy_diff:.2f}, score={energy_score:.2f})"
    )

    # --- Acousticness score (10%) — distance-based ---
    target_acoustic = 1.0 if user_prefs['likes_acoustic'] else 0.0
    acoustic_diff = abs(target_acoustic - song['acousticness'])
    acousticness_score = 10 - (10 * acoustic_diff)
    reasons.append(
        f"Acousticness {song['acousticness']:.2f} vs target {target_acoustic:.1f} "
        f"(diff={acoustic_diff:.2f}, score={acousticness_score:.2f})"
    )

    # --- Popularity score (10%) — higher popularity → higher score ---
    popularity = song.get('popularity', 50)
    popularity_score = popularity / 10.0  # 0–100 scaled to 0–10
    min_pop = user_prefs.get('min_popularity', 0)
    if popularity < min_pop:
        popularity_score = max(0, popularity_score - 3)
        reasons.append(
            f"Popularity {popularity}/100 below your minimum {min_pop} "
            f"(penalised, score={popularity_score:.1f})"
        )
    else:
        reasons.append(f"Popularity {popularity}/100 (score={popularity_score:.1f})")

    # --- Release decade score (10%) — era preference ---
    preferred_decade = user_prefs.get('preferred_decade')
    song_decade = song.get('release_decade', 2010)
    if preferred_decade is not None:
        decade_diff = abs(preferred_decade - song_decade) / 10  # each decade apart = 1 step
        decade_score = max(0.0, 10.0 - decade_diff * 2)
        reasons.append(
            f"Release decade {song_decade}s vs preferred {preferred_decade}s "
            f"(diff={int(decade_diff)} step(s), score={decade_score:.1f})"
        )
    else:
        decade_score = 5.0  # neutral when no era preference set
        reasons.append(f"Release decade {song_decade}s (no preference, score=5.0)")

    # --- Mood tags score (10%) — tag overlap ---
    preferred_tags = user_prefs.get('preferred_mood_tags', [])
    song_tags_raw = song.get('mood_tags', '')
    song_tags = [t.strip() for t in song_tags_raw.split('|') if t.strip()]
    if preferred_tags and song_tags:
        overlap = len(set(preferred_tags) & set(song_tags))
        mood_tags_score = min(10.0, overlap * 10.0 / len(preferred_tags))
        reasons.append(
            f"Mood tags {song_tags} vs preferred {preferred_tags}: "
            f"{overlap} match(es) (score={mood_tags_score:.1f})"
        )
    else:
        mood_tags_score = 5.0  # neutral when no tags set
        reasons.append(f"Mood tags {song_tags} (no preference, score=5.0)")

    # --- Weighted base score (weights come from the chosen strategy) ---
    base_score = (
        genre_score        * weights["genre"] +
        mood_score         * weights["mood"] +
        energy_score       * weights["energy"] +
        acousticness_score * weights["acousticness"] +
        popularity_score   * weights["popularity"] +
        decade_score       * weights["decade"] +
        mood_tags_score    * weights["mood_tags"]
    )

    # --- Explicit content penalty ---
    is_explicit = bool(song.get('explicit', 0))
    allow_explicit = user_prefs.get('allow_explicit', True)
    if is_explicit and not allow_explicit:
        base_score = max(0.0, base_score - 3.0)
        reasons.append("Explicit content (filtered by preference, -3)")

    # --- Feedback adjustment (+/-2) ---
    song_id = song.get('id')
    if song_id in user_prefs.get('likes', []):
        adjustment = 2
        reasons.append("You liked this song before (+2)")
    elif song_id in user_prefs.get('skips', []):
        adjustment = -2
        reasons.append("You skipped this song before (-2)")
    else:
        adjustment = 0

    final_score = max(0.0, min(10.0, base_score + adjustment))
    return (final_score, reasons)

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
    max_per_artist: int = 2,
    max_per_genre: int = 3,
) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    strategy       — "balanced" | "genre_first" | "mood_first"
    max_per_artist — cap on songs from the same artist in the top-k
    max_per_genre  — cap on songs from the same genre in the top-k
    """
    scored = [(song, *score_song(user_prefs, song, strategy)) for song in songs]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    diverse = diversity_rerank(ranked, k, max_per_artist, max_per_genre)
    return [(song, score, "\n".join(reasons)) for song, score, reasons in diverse]
