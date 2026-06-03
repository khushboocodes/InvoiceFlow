"""Dealer and asset masters: loaders + offline mining utilities.

The ``Masters`` object provides canonical lists of dealer names and tractor
models that the normalization stage uses for fuzzy / exact matching. The
training-time mining functions build these masters from the unlabeled
training set itself, so the pipeline doesn't depend on any external
authoritative master file.

Validates Requirements: 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3, 13.4
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DealerEntry:
    """A canonical dealer name plus alternate spellings observed in the data."""

    canonical: str
    aliases: tuple[str, ...]
    frequency: int

    def all_forms(self) -> tuple[str, ...]:
        """Every string we'll fuzzy-match against."""
        return (self.canonical,) + self.aliases


@dataclass(frozen=True)
class AssetEntry:
    """A tractor (brand, model) record."""

    brand: str
    model: str
    full_name: str  # ``"<brand> <model>"`` — the canonical exact-match key


@dataclass
class Masters:
    """Container for both masters."""

    dealer: list[DealerEntry] = field(default_factory=list)
    asset: list[AssetEntry] = field(default_factory=list)

    @property
    def brand_keywords(self) -> tuple[str, ...]:
        """Distinct brand names — used by the Tier-1 model_name extractor."""
        return tuple(sorted({entry.brand for entry in self.asset}))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load(data_dir: Path) -> Masters:
    """Load ``dealer_master.json`` and ``asset_master.json`` from ``data_dir``.

    Missing files are tolerated (returns an empty Masters); malformed files
    raise ``ValueError`` with the offending key path.
    """
    dealer = _load_dealer_master(data_dir / "dealer_master.json")
    asset = _load_asset_master(data_dir / "asset_master.json")
    return Masters(dealer=dealer, asset=asset)


def _load_dealer_master(path: Path) -> list[DealerEntry]:
    if not path.exists():
        logger.warning("Dealer master not found at %s; returning empty list", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed dealer master JSON at {path}: {exc}") from exc
    entries_raw = data.get("entries", [])
    out: list[DealerEntry] = []
    for i, raw in enumerate(entries_raw):
        try:
            out.append(
                DealerEntry(
                    canonical=str(raw["canonical"]),
                    aliases=tuple(str(a) for a in raw.get("aliases", [])),
                    frequency=int(raw.get("frequency", 1)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed dealer entry at index {i}: {exc}") from exc
    return out


def _load_asset_master(path: Path) -> list[AssetEntry]:
    if not path.exists():
        logger.warning("Asset master not found at %s; returning empty list", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed asset master JSON at {path}: {exc}") from exc
    entries_raw = data.get("entries", [])
    out: list[AssetEntry] = []
    for i, raw in enumerate(entries_raw):
        try:
            out.append(
                AssetEntry(
                    brand=str(raw["brand"]),
                    model=str(raw["model"]),
                    full_name=str(raw.get("full_name", f"{raw['brand']} {raw['model']}")),
                )
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed asset entry at index {i}: {exc}") from exc
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_dealer_master(entries: Iterable[DealerEntry], path: Path) -> None:
    """Write a canonical, byte-stable dealer master JSON file.

    Property 9: re-running the miner on the same input must produce a
    byte-identical output. We sort entries and aliases lexicographically
    and emit with stable JSON formatting to satisfy that contract.
    """
    sorted_entries = sorted(entries, key=lambda e: e.canonical)
    payload = {
        "version": 1,
        "entries": [
            {
                "canonical": e.canonical,
                "aliases": sorted(e.aliases),
                "frequency": e.frequency,
            }
            for e in sorted_entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def save_asset_master(entries: Iterable[AssetEntry], path: Path) -> None:
    sorted_entries = sorted(entries, key=lambda e: (e.full_name, e.brand))
    payload = {
        "version": 1,
        "entries": [
            {"brand": e.brand, "model": e.model, "full_name": e.full_name}
            for e in sorted_entries
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Mining helpers — pure logic. The actual OCR-driven miner lives in
# scripts/mine_masters.py to keep this module free of heavy imports.
# --------------------------------------------------------------------------- #


def cluster_dealer_candidates(
    candidates: Iterable[str], *, similarity_threshold: int = 85
) -> list[DealerEntry]:
    """Cluster noisy dealer-name strings into canonical entries.

    Uses RapidFuzz token-set ratio with greedy single-linkage. The most
    frequent string in each cluster becomes the canonical, others become
    aliases. Singletons (frequency < 2) are dropped — they're typically
    OCR noise.

    Args:
        candidates: Iterable of raw dealer-name strings (already
            normalized, i.e. whitespace-collapsed).
        similarity_threshold: RapidFuzz token-set ratio cutoff in [0, 100].

    Returns:
        A list of :class:`DealerEntry` records, sorted by canonical.
    """
    from rapidfuzz import fuzz, process

    counts: dict[str, int] = {}
    for cand in candidates:
        normalized = " ".join(cand.split()).strip()
        if not normalized or len(normalized) < 4:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1

    if not counts:
        return []

    unique_strings = list(counts.keys())
    visited: set[str] = set()
    clusters: list[list[str]] = []

    for s in unique_strings:
        if s in visited:
            continue
        # Find every other string within the similarity threshold.
        cluster = [s]
        visited.add(s)
        matches = process.extract(
            s, unique_strings, scorer=fuzz.token_set_ratio, score_cutoff=similarity_threshold
        )
        for match_str, score, _ in matches:
            if match_str != s and match_str not in visited:
                cluster.append(match_str)
                visited.add(match_str)
        clusters.append(cluster)

    entries: list[DealerEntry] = []
    for cluster in clusters:
        cluster_total = sum(counts[c] for c in cluster)
        if cluster_total < 2:
            continue
        # Most-frequent string wins canonicality; ties broken by length (longer
        # is usually more informative).
        cluster.sort(key=lambda c: (counts[c], len(c)), reverse=True)
        canonical = cluster[0]
        aliases = tuple(c for c in cluster[1:] if c != canonical)
        entries.append(DealerEntry(canonical=canonical, aliases=aliases, frequency=cluster_total))

    entries.sort(key=lambda e: e.canonical)
    return entries


# --------------------------------------------------------------------------- #
# Asset master seeding
# --------------------------------------------------------------------------- #

# Curated list of common Indian tractor brands. Used as the seed for asset
# master mining and as a fallback brand-keyword list for the Tier-1
# extractor when no master file is bundled.
SEED_BRANDS: tuple[str, ...] = (
    "Mahindra",
    "Sonalika",
    "John Deere",
    "Massey Ferguson",
    "Swaraj",
    "New Holland",
    "Eicher",
    "Powertrac",
    "Farmtrac",
    "Kubota",
    "Force",
    "HMT",
    "TAFE",
    "Captain",
    "Indo Farm",
    "Preet",
    "Standard",
    "VST",
)


def merge_asset_entries(
    mined: Iterable[AssetEntry], curated: Iterable[AssetEntry]
) -> list[AssetEntry]:
    """Union mined entries with a hand-curated seed list, deduped on full_name."""
    seen: set[str] = set()
    out: list[AssetEntry] = []
    for entry in list(curated) + list(mined):
        key = entry.full_name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    out.sort(key=lambda e: (e.full_name, e.brand))
    return out
