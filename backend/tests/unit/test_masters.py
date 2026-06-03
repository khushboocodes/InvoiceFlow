"""Unit tests for utils.masters — dealer + asset master loaders and miners.

Validates Requirements: 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3, 13.4
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.masters import (
    SEED_BRANDS,
    AssetEntry,
    DealerEntry,
    Masters,
    cluster_dealer_candidates,
    load,
    merge_asset_entries,
    save_asset_master,
    save_dealer_master,
)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_load_returns_empty_masters_when_files_missing(tmp_path: Path):
    masters = load(tmp_path)
    assert masters.dealer == []
    assert masters.asset == []


def test_load_parses_well_formed_files(tmp_path: Path):
    (tmp_path / "dealer_master.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "canonical": "ABC TRACTORS PVT LTD",
                        "aliases": ["ABC Tractors", "ABC TRAC"],
                        "frequency": 5,
                    }
                ],
            }
        )
    )
    (tmp_path / "asset_master.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {"brand": "Mahindra", "model": "575 DI", "full_name": "Mahindra 575 DI"}
                ],
            }
        )
    )
    masters = load(tmp_path)
    assert len(masters.dealer) == 1
    assert masters.dealer[0].canonical == "ABC TRACTORS PVT LTD"
    assert "ABC Tractors" in masters.dealer[0].aliases
    assert len(masters.asset) == 1
    assert masters.asset[0].brand == "Mahindra"


def test_load_raises_on_malformed_dealer_master(tmp_path: Path):
    (tmp_path / "dealer_master.json").write_text("not valid json {")
    with pytest.raises(ValueError, match="Malformed"):
        load(tmp_path)


def test_brand_keywords_dedupes_and_sorts():
    masters = Masters(
        asset=[
            AssetEntry(brand="Mahindra", model="575", full_name="Mahindra 575"),
            AssetEntry(brand="Sonalika", model="DI 60", full_name="Sonalika DI 60"),
            AssetEntry(brand="Mahindra", model="265", full_name="Mahindra 265"),
        ]
    )
    assert masters.brand_keywords == ("Mahindra", "Sonalika")


# --------------------------------------------------------------------------- #
# Mining: cluster_dealer_candidates
# --------------------------------------------------------------------------- #


def test_cluster_dealer_candidates_merges_near_duplicates():
    """Three near-duplicate dealer names should collapse into one entry."""
    candidates = [
        "MADHU PAVAN AUTOMOBILES",
        "MADHU PAVAN AUTOMOBILES",
        "MADHU PAVAN AUTOMOBILES",  # duplicate of canonical
        "MADHU PAWAN AUTOMOBILES",  # OCR typo (V → W)
    ]
    entries = cluster_dealer_candidates(candidates)
    assert len(entries) == 1
    assert entries[0].canonical == "MADHU PAVAN AUTOMOBILES"
    assert "MADHU PAWAN AUTOMOBILES" in entries[0].aliases
    assert entries[0].frequency == 4


def test_cluster_dealer_candidates_drops_singletons():
    """Singletons (frequency < 2) are likely OCR noise."""
    candidates = ["ABC TRACTORS", "XYZ MOTORS"]
    entries = cluster_dealer_candidates(candidates)
    assert entries == []


def test_cluster_dealer_candidates_handles_empty_input():
    assert cluster_dealer_candidates([]) == []


def test_cluster_dealer_candidates_keeps_distinct_dealers_separate():
    candidates = [
        "MADHU PAVAN AUTOMOBILES",
        "MADHU PAVAN AUTOMOBILES",
        "SRI AMUTHAM TRACTORS",
        "SRI AMUTHAM TRACTORS",
    ]
    entries = cluster_dealer_candidates(candidates)
    canonicals = {e.canonical for e in entries}
    assert canonicals == {"MADHU PAVAN AUTOMOBILES", "SRI AMUTHAM TRACTORS"}


# --------------------------------------------------------------------------- #
# Property 9: master mining stability
# --------------------------------------------------------------------------- #


def test_save_dealer_master_is_byte_stable(tmp_path: Path):
    """Re-running the miner on identical input must produce identical output."""
    entries = [
        DealerEntry(canonical="ABC TRACTORS", aliases=("ABC Trac", "ABC"), frequency=3),
        DealerEntry(canonical="XYZ MOTORS", aliases=(), frequency=2),
    ]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    save_dealer_master(entries, a)
    # Reverse the input order — output must be identical.
    save_dealer_master(list(reversed(entries)), b)
    assert a.read_bytes() == b.read_bytes()


def test_save_asset_master_is_byte_stable(tmp_path: Path):
    entries = [
        AssetEntry(brand="Mahindra", model="575 DI", full_name="Mahindra 575 DI"),
        AssetEntry(brand="Sonalika", model="DI 60", full_name="Sonalika DI 60"),
    ]
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    save_asset_master(entries, a)
    save_asset_master(list(reversed(entries)), b)
    assert a.read_bytes() == b.read_bytes()


# --------------------------------------------------------------------------- #
# Asset master seeding
# --------------------------------------------------------------------------- #


def test_seed_brands_includes_major_indian_tractor_brands():
    """Sanity check — common brands present so model extraction can fall back to
    keyword detection when no master is loaded."""
    assert "Mahindra" in SEED_BRANDS
    assert "Sonalika" in SEED_BRANDS
    assert "John Deere" in SEED_BRANDS
    assert "Massey Ferguson" in SEED_BRANDS


def test_merge_asset_entries_dedupes_on_full_name():
    curated = [AssetEntry(brand="Mahindra", model="575", full_name="Mahindra 575")]
    mined = [
        AssetEntry(brand="Mahindra", model="575", full_name="Mahindra 575"),  # duplicate
        AssetEntry(brand="Sonalika", model="DI 60", full_name="Sonalika DI 60"),
    ]
    merged = merge_asset_entries(mined, curated)
    full_names = [e.full_name for e in merged]
    assert full_names == sorted(full_names)
    assert len(merged) == 2  # curated + new mined, duplicate dropped


def test_merge_asset_entries_handles_case_difference_as_dupe():
    curated = [AssetEntry(brand="Mahindra", model="575", full_name="Mahindra 575")]
    mined = [AssetEntry(brand="MAHINDRA", model="575", full_name="MAHINDRA 575")]
    merged = merge_asset_entries(mined, curated)
    assert len(merged) == 1
