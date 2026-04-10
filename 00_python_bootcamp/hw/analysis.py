"""HW00 — Analysis: Songs Dataset

Implement the five functions below using pandas. Each function has a docstring
describing what it should do and showing examples.

After completing the functions, run this script to print your results:

    uv run python analysis.py

Use the printed output to answer the questions in writeup.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_songs(path: Path) -> pd.DataFrame:
    """Load the songs CSV and return a clean DataFrame.

    Casts year, weeks_on_chart, and peak_position to int.
    Casts streams_millions to float.

    Args:
        path: Path to the songs CSV file.

    Returns:
        A DataFrame with the correct column dtypes.

    Example:
        >>> df = load_songs(Path("data/songs.csv"))
        >>> df.dtypes["year"]
        dtype('int64')
    """
    pass


def top_charting_songs(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top n songs by streams_millions, sorted in descending order.

    Args:
        df: The songs DataFrame.
        n: Number of songs to return (default 10).

    Returns:
        A DataFrame with the top n rows by streams_millions, highest first.

    Example:
        >>> top = top_charting_songs(df, n=3)
        >>> len(top)
        3
        >>> top["streams_millions"].is_monotonic_decreasing
        True
    """
    pass


def avg_weeks_by_genre(df: pd.DataFrame) -> dict[str, float]:
    """Return the average weeks_on_chart for each genre.

    Args:
        df: The songs DataFrame.

    Returns:
        A plain Python dict mapping genre name (str) to average weeks (float).

    Example:
        >>> avgs = avg_weeks_by_genre(df)
        >>> isinstance(avgs, dict)
        True
    """
    pass


def most_streamed_artist(df: pd.DataFrame) -> str:
    """Return the name of the artist with the highest total streams_millions.

    If an artist has multiple songs, sum all their streams.

    Args:
        df: The songs DataFrame.

    Returns:
        The artist name as a string.

    Example:
        >>> artist = most_streamed_artist(df)
        >>> isinstance(artist, str)
        True
    """
    pass


def hits_per_year(df: pd.DataFrame, max_position: int = 10) -> dict[int, int]:
    """Count songs with peak_position <= max_position, grouped by year.

    A "hit" is any song that reached position max_position or better (lower number).

    Args:
        df: The songs DataFrame.
        max_position: Peak position threshold (default 10).

    Returns:
        A plain Python dict mapping year (int) to hit count (int).

    Example:
        >>> hits = hits_per_year(df, max_position=5)
        >>> all(isinstance(k, int) for k in hits.keys())
        True
    """
    pass


# ── Main: print results for writeup.md ────────────────────────────────────────

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data" / "songs.csv"
    df = load_songs(data_path)

    print("=== Top 10 Songs by Streams ===")
    top = top_charting_songs(df, n=10)
    for _, row in top.iterrows():
        print(
            f"  {row['title']} — {row['artist']} ({row['streams_millions']:.0f}M streams)"
        )

    print("\n=== Average Weeks on Chart by Genre ===")
    avg_weeks = avg_weeks_by_genre(df)
    for genre, avg in sorted(avg_weeks.items(), key=lambda x: -x[1]):
        print(f"  {genre}: {avg:.1f} weeks")

    print("\n=== Most Streamed Artist ===")
    artist = most_streamed_artist(df)
    total_streams = df.groupby("artist")["streams_millions"].sum()[artist]
    print(f"  {artist} ({total_streams:.0f}M total streams)")

    print("\n=== Top-10 Hits per Year (peak position <= 10) ===")
    hits = hits_per_year(df, max_position=10)
    for year, count in sorted(hits.items()):
        print(f"  {year}: {count} hit(s)")
