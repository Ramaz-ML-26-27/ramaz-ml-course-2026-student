"""Tests for HW00 Part 2 — analysis.py

Tests use synthetic DataFrames instead of the real songs.csv so that:
  - Tests check your logic, not whether you hardcoded the right answer.
  - Tests are fast and self-contained.

Run with: uv run pytest -m analysis
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis import (
    avg_weeks_by_genre,
    hits_per_year,
    load_songs,
    most_streamed_artist,
    top_charting_songs,
)

# ── Shared fixture ─────────────────────────────────────────────────────────────

SAMPLE: dict[str, list] = {
    "title": [
        "Song A",
        "Song B",
        "Song C",
        "Song D",
        "Song E",
        "Song F",
    ],
    "artist": [
        "Artist X",
        "Artist Y",
        "Artist X",
        "Artist Z",
        "Artist Y",
        "Artist X",
    ],
    "genre": [
        "Pop",
        "Rock",
        "Pop",
        "Hip-Hop",
        "Rock",
        "Pop",
    ],
    "year": [2021, 2021, 2022, 2022, 2023, 2023],
    "weeks_on_chart": [20, 10, 30, 15, 25, 5],
    "peak_position": [1, 5, 2, 8, 12, 3],
    "streams_millions": [1000.0, 500.0, 800.0, 300.0, 600.0, 400.0],
}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(SAMPLE)


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.analysis
class TestLoadSongs:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "songs.csv"
        csv_path.write_text(
            "title,artist,genre,year,weeks_on_chart,peak_position,streams_millions\n"
            "Song A,Artist X,Pop,2021,20,1,1000.0\n"
        )
        df = load_songs(csv_path)
        assert isinstance(df, pd.DataFrame), (
            f"load_songs should return a DataFrame; got {type(df).__name__}"
        )

    def test_year_is_int(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "songs.csv"
        csv_path.write_text(
            "title,artist,genre,year,weeks_on_chart,peak_position,streams_millions\n"
            "Song A,Artist X,Pop,2021,20,1,1000.0\n"
        )
        df = load_songs(csv_path)
        assert pd.api.types.is_integer_dtype(df["year"]), (
            f"'year' should be int dtype; got {df['year'].dtype}"
        )

    def test_weeks_on_chart_is_int(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "songs.csv"
        csv_path.write_text(
            "title,artist,genre,year,weeks_on_chart,peak_position,streams_millions\n"
            "Song A,Artist X,Pop,2021,20,1,1000.0\n"
        )
        df = load_songs(csv_path)
        assert pd.api.types.is_integer_dtype(df["weeks_on_chart"]), (
            f"'weeks_on_chart' should be int dtype; got {df['weeks_on_chart'].dtype}"
        )

    def test_peak_position_is_int(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "songs.csv"
        csv_path.write_text(
            "title,artist,genre,year,weeks_on_chart,peak_position,streams_millions\n"
            "Song A,Artist X,Pop,2021,20,1,1000.0\n"
        )
        df = load_songs(csv_path)
        assert pd.api.types.is_integer_dtype(df["peak_position"]), (
            f"'peak_position' should be int dtype; got {df['peak_position'].dtype}"
        )

    def test_streams_is_float(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "songs.csv"
        csv_path.write_text(
            "title,artist,genre,year,weeks_on_chart,peak_position,streams_millions\n"
            "Song A,Artist X,Pop,2021,20,1,1000.0\n"
        )
        df = load_songs(csv_path)
        assert pd.api.types.is_float_dtype(df["streams_millions"]), (
            f"'streams_millions' should be float dtype; got {df['streams_millions'].dtype}"
        )


@pytest.mark.analysis
class TestTopChartingSongs:
    def test_returns_n_rows(self, sample_df: pd.DataFrame) -> None:
        result = top_charting_songs(sample_df, n=3)
        assert len(result) == 3, (
            f"top_charting_songs(df, n=3) should return 3 rows; got {len(result)}"
        )

    def test_sorted_descending(self, sample_df: pd.DataFrame) -> None:
        result = top_charting_songs(sample_df, n=4)
        streams = result["streams_millions"].tolist()
        assert streams == sorted(streams, reverse=True), (
            f"Rows should be sorted by streams_millions descending; got {streams}"
        )

    def test_top_song_is_highest(self, sample_df: pd.DataFrame) -> None:
        result = top_charting_songs(sample_df, n=1)
        # Song A has 1000.0 — the highest in SAMPLE
        assert result.iloc[0]["streams_millions"] == 1000.0, (
            f"Top song should have 1000.0 streams; got {result.iloc[0]['streams_millions']}"
        )

    def test_default_n_is_10(self, sample_df: pd.DataFrame) -> None:
        # sample_df has 6 rows; default n=10 should return all 6
        result = top_charting_songs(sample_df)
        assert len(result) == min(10, len(sample_df)), (
            f"Default n=10 with {len(sample_df)}-row df should return all rows; got {len(result)}"
        )


@pytest.mark.analysis
class TestAvgWeeksByGenre:
    def test_returns_dict(self, sample_df: pd.DataFrame) -> None:
        result = avg_weeks_by_genre(sample_df)
        assert isinstance(result, dict), (
            f"avg_weeks_by_genre should return a dict; got {type(result).__name__}"
        )

    def test_all_genres_present(self, sample_df: pd.DataFrame) -> None:
        result = avg_weeks_by_genre(sample_df)
        assert set(result.keys()) == {"Pop", "Rock", "Hip-Hop"}, (
            f"Expected genres Pop, Rock, Hip-Hop; got {set(result.keys())}"
        )

    def test_pop_average(self, sample_df: pd.DataFrame) -> None:
        # Pop songs: 20, 30, 5 → avg = 55/3 ≈ 18.33
        result = avg_weeks_by_genre(sample_df)
        expected = (20 + 30 + 5) / 3
        assert abs(result["Pop"] - expected) < 1e-6, (
            f"Pop avg weeks: expected {expected:.4f}; got {result['Pop']}"
        )

    def test_rock_average(self, sample_df: pd.DataFrame) -> None:
        # Rock songs: 10, 25 → avg = 17.5
        result = avg_weeks_by_genre(sample_df)
        assert abs(result["Rock"] - 17.5) < 1e-6, (
            f"Rock avg weeks: expected 17.5; got {result['Rock']}"
        )

    def test_hiphop_average(self, sample_df: pd.DataFrame) -> None:
        # Hip-Hop songs: 15 → avg = 15.0
        result = avg_weeks_by_genre(sample_df)
        assert abs(result["Hip-Hop"] - 15.0) < 1e-6, (
            f"Hip-Hop avg weeks: expected 15.0; got {result['Hip-Hop']}"
        )


@pytest.mark.analysis
class TestMostStreamedArtist:
    def test_returns_string(self, sample_df: pd.DataFrame) -> None:
        result = most_streamed_artist(sample_df)
        assert isinstance(result, str), (
            f"most_streamed_artist should return a string; got {type(result).__name__}"
        )

    def test_correct_artist(self, sample_df: pd.DataFrame) -> None:
        # Artist X: 1000 + 800 + 400 = 2200
        # Artist Y: 500 + 600 = 1100
        # Artist Z: 300
        result = most_streamed_artist(sample_df)
        assert result == "Artist X", (
            f"Artist X has 2200M total streams (highest); got {result!r}"
        )

    def test_uses_total_not_single(self, sample_df: pd.DataFrame) -> None:
        # Verify Artist Y (max single-song: 600) loses to Artist X (total: 2200)
        result = most_streamed_artist(sample_df)
        assert result != "Artist Y", (
            "most_streamed_artist should sum all songs per artist, not take the single highest"
        )

    def test_custom_data(self) -> None:
        df = pd.DataFrame(
            {
                "artist": ["Solo", "Multi", "Multi"],
                "streams_millions": [900.0, 500.0, 600.0],
                "title": ["A", "B", "C"],
                "genre": ["Pop", "Pop", "Pop"],
                "year": [2022, 2022, 2022],
                "weeks_on_chart": [10, 10, 10],
                "peak_position": [1, 2, 3],
            }
        )
        result = most_streamed_artist(df)
        # Multi: 500+600=1100 > Solo: 900
        assert result == "Multi", (
            f"Multi has 1100M total vs Solo's 900M; got {result!r}"
        )


@pytest.mark.analysis
class TestHitsPerYear:
    def test_returns_dict(self, sample_df: pd.DataFrame) -> None:
        result = hits_per_year(sample_df)
        assert isinstance(result, dict), (
            f"hits_per_year should return a dict; got {type(result).__name__}"
        )

    def test_correct_counts_default(self, sample_df: pd.DataFrame) -> None:
        # peak_position <= 10: Song A(1), Song B(5), Song C(2), Song D(8), Song F(3)
        # Song E has peak_position=12 — excluded
        # 2021: A(1), B(5) = 2 hits
        # 2022: C(2), D(8) = 2 hits
        # 2023: F(3) = 1 hit  [E is excluded]
        result = hits_per_year(sample_df, max_position=10)
        assert result.get(2021) == 2, (
            f"2021 should have 2 hits (positions 1 and 5); got {result.get(2021)}"
        )
        assert result.get(2022) == 2, (
            f"2022 should have 2 hits (positions 2 and 8); got {result.get(2022)}"
        )
        assert result.get(2023) == 1, (
            f"2023 should have 1 hit (position 3 only; position 12 is excluded); "
            f"got {result.get(2023)}"
        )

    def test_strict_cutoff(self, sample_df: pd.DataFrame) -> None:
        # max_position=5: only positions 1, 5, 2, 3 qualify (not 8, 12)
        result = hits_per_year(sample_df, max_position=5)
        assert result.get(2021) == 2, (
            f"2021 with max_position=5: positions 1 and 5 both qualify; got {result.get(2021)}"
        )
        assert result.get(2022) == 1, (
            f"2022 with max_position=5: only position 2 qualifies (8 is excluded); "
            f"got {result.get(2022)}"
        )

    def test_year_with_no_hits_excluded(self, sample_df: pd.DataFrame) -> None:
        # max_position=1: only Song A (position 1) qualifies → 2021 only
        result = hits_per_year(sample_df, max_position=1)
        assert 2021 in result, "2021 has a song at position 1 and should appear"
        assert 2022 not in result, (
            "2022 has no songs at position 1 and should not appear in result"
        )
        assert 2023 not in result, (
            "2023 has no songs at position 1 and should not appear in result"
        )
