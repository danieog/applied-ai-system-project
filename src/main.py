"""
Command line runner for the Music Recommender Simulation.

Displays top recommendations in a formatted ASCII table with a
per-song score breakdown below.  Install `tabulate` for an even
nicer table:
    pip install tabulate
"""

from recommender import load_songs, Recommender, UserProfile, recommend_songs
from rag_recommender import RAGRecommender

# ── Optional tabulate for prettier tables ─────────────────────────────────
try:
    from tabulate import tabulate as _tabulate  # type: ignore[import-untyped]
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


# ── Small visual helpers ──────────────────────────────────────────────────

def _stars(score: float) -> str:
    """0-10 score → 5-star unicode string."""
    filled = round(score / 2)
    return "★" * filled + "☆" * (5 - filled)


def _bar(score: float, width: int = 10) -> str:
    """0-10 score → filled block bar."""
    filled = round(score / 10 * width)
    return "█" * filled + "░" * (width - filled)


# ── Layout helpers ────────────────────────────────────────────────────────

def _print_header(label: str, strategy: str, total: int, k: int) -> None:
    line1 = f"  ♫  {label}"
    line2 = f"  Strategy: {strategy}  ·  top {k} of {total} songs"
    width = max(len(line1), len(line2)) + 4
    print()
    print("╔" + "═" * width + "╗")
    print("║" + line1.ljust(width) + "║")
    print("║" + line2.ljust(width) + "║")
    print("╚" + "═" * width + "╝")


def _print_table(rows: list) -> None:
    """
    Print a formatted results table.
    rows: list of 7-tuples — (#, Title, Artist, Genre, Score, Rating, Bar)
    """
    headers = ("#", "Title", "Artist", "Genre", "Score", "Rating", "")

    if _HAS_TABULATE:
        print()
        print(_tabulate(rows, headers=headers, tablefmt="rounded_outline",
                        colalign=("right", "left", "left", "left",
                                  "right", "left", "left")))
        return

    # ── Pure-Python fallback ──────────────────────────────────────────────
    # Compute column widths from headers + data
    all_rows = [headers] + [tuple(str(c) for c in r) for r in rows]
    col_w = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    def _fmt_row(row) -> str:
        cells = [str(row[i]).ljust(col_w[i]) for i in range(len(headers))]
        # Right-align the rank (#) and score columns
        cells[0] = str(row[0]).rjust(col_w[0])
        cells[4] = str(row[4]).rjust(col_w[4])
        return "  │  ".join(cells)

    separator = "──┼──".join("─" * w for w in col_w)

    print()
    print("  " + _fmt_row(headers))
    print("  " + separator)
    for r in rows:
        print("  " + _fmt_row(r))
    print("  " + separator)


def _print_breakdown(recs: list) -> None:
    """Print a per-song score reason breakdown beneath the table."""
    width = 64
    print()
    print("  ┌" + "─" * width + "┐")
    print("  │" + "  Score Breakdown".ljust(width) + "│")
    print("  └" + "─" * width + "┘")

    for i, (song, score, explanation) in enumerate(recs, 1):
        title_line = f"  {i}.  {song['title']}  —  {score:.2f} / 10  {_stars(score)}"
        print()
        print(title_line)
        print("      " + "·" * (len(title_line) - 6))
        for line in explanation.split("\n"):
            print(f"      {line}")


# ── Public entry point ────────────────────────────────────────────────────

def run_profile(
    label: str,
    user_prefs: dict,
    songs: list,
    k: int = 5,
    strategy: str = "balanced",
) -> None:
    """Print a formatted recommendations block for one user profile."""
    recs = recommend_songs(user_prefs, songs, k=k, strategy=strategy)
    if not recs:
        print(f"\n[{label}]  No recommendations returned.")
        return

    _print_header(label, strategy, len(songs), k)

    table_rows = [
        (
            i,
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.2f}",
            _stars(score),
            _bar(score),
        )
        for i, (song, score, _) in enumerate(recs, 1)
    ]
    _print_table(table_rows)
    _print_breakdown(recs)
    print()


# ── Profiles ──────────────────────────────────────────────────────────────

def main() -> None:
    songs = load_songs("../data/songs.csv")
    print(f"\nLoaded {len(songs)} songs.")

    danieog = {
        "favorite_genre": "indie pop",
        "favorite_mood": "chill",
        "target_energy": 0.73,
        "likes_acoustic": False,
        "likes": [13, 18, 8],
        "skips": [12, 16, 3],
    }
    run_profile("My Music  ·  balanced", danieog, songs, strategy="balanced")
    run_profile("My Music  ·  genre first", danieog, songs, strategy="genre_first")
    run_profile("My Music  ·  mood first", danieog, songs, strategy="mood_first")


if __name__ == "__main__":
    main()
