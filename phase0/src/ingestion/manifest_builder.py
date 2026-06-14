"""
phase0/src/ingestion/manifest_builder.py
Builds a structured CSV manifest from an AnnotationAdapter and cross-references
the optional ``all_folder`` directory.

Public API:
    ManifestBuilder(config, project_root).build(adapter)  → pd.DataFrame
    load_config(config_path)                              → dict
    get_project_root()                                    → Path
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from phase0.src.ingestion.base import AnnotationAdapter
from phase0.src.utils.io import compute_sha256, get_image_format
from phase0.src.utils.logging import get_logger


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Parse a YAML config file and return its contents as a dictionary.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a plain Python ``dict``.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError:    If the file cannot be parsed.
    """
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_project_root() -> Path:
    """Return the project root directory (two parents above this file's location).

    This file lives at ``phase0/src/ingestion/manifest_builder.py``, so:
        __file__ → …/phase0/src/ingestion/manifest_builder.py
        .parent   → …/phase0/src/ingestion/
        .parent   → …/phase0/src/
        .parent   → …/phase0/
        .parent   → …/  ← project root

    Returns:
        Absolute :class:`~pathlib.Path` to the project root.
    """
    return Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# ManifestBuilder
# ---------------------------------------------------------------------------

class ManifestBuilder:
    """Orchestrates dataset ingestion and saves manifests + logs to disk.

    Args:
        config:       Full project config dictionary (loaded from phase0.yaml).
        project_root: Absolute path to the project root directory.
    """

    def __init__(self, config: dict, project_root: Path) -> None:
        self.config: dict = config
        self.project_root: Path = Path(project_root)
        self._logger = get_logger(__name__)

        # Resolve paths from config
        ingestion_cfg: dict = config.get("ingestion", {})
        paths_cfg: dict = config.get("paths", {})

        self.data_root: Path = Path(ingestion_cfg.get("data_root", "data"))
        if not self.data_root.is_absolute():
            self.data_root = self.project_root / self.data_root

        self.outputs_root: Path = Path(paths_cfg.get("outputs_root", "phase0/outputs"))
        if not self.outputs_root.is_absolute():
            self.outputs_root = self.project_root / self.outputs_root
        self.outputs_root.mkdir(parents=True, exist_ok=True)

        # Severity label → human-readable name
        ingestion_cfg = config.get("ingestion", {})
        self.severity_map: dict[int, str] = {
            int(k): v
            for k, v in ingestion_cfg.get("severity_map", config.get("severity_map", {
                0: "mild",
                1: "moderate",
                2: "severe",
                3: "very_severe",
            })).items()
        }

        # Optional all_folder name
        self._all_folder_name: str = ingestion_cfg.get("all_folder", "")

        # Populated by _cross_reference_all_folder for use in _save_ingestion_log
        self._all_folder_summary: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public build method
    # ------------------------------------------------------------------

    def build(self, adapter: AnnotationAdapter) -> pd.DataFrame:
        """Run full ingestion and produce ``manifest.csv``.

        Steps:
            1. Parse all :class:`~phase0.src.ingestion.base.ImageRecord` objects.
            2. Compute file metadata (SHA-256, size, format, existence).
            3. Detect duplicate images (same SHA-256).
            4. Save ``manifest.csv``.
            5. Cross-reference ``all_folder`` directory (if configured).
            6. Save ``ingestion_log.json``.

        Args:
            adapter: A validated :class:`~phase0.src.ingestion.base.AnnotationAdapter`.

        Returns:
            A :class:`~pandas.DataFrame` with one row per image record.
        """
        self._logger.info(
            "ManifestBuilder.build() started. Adapter format: '%s'.",
            adapter.format_name,
        )

        rows: list[dict] = []
        seen_sha256: dict[str, str] = {}   # sha256 → first image_id seen

        for record in adapter.parse():
            path = record.image_path
            exists = path.exists()

            if exists:
                try:
                    sha256 = compute_sha256(path)
                except OSError:
                    sha256 = ""
                try:
                    size_bytes: int | None = path.stat().st_size
                except OSError:
                    size_bytes = None
                fmt = get_image_format(path)
            else:
                sha256 = ""
                size_bytes = None
                fmt = get_image_format(path)

            # Duplicate detection
            is_duplicate = False
            if sha256 and sha256 in seen_sha256:
                is_duplicate = True
            elif sha256:
                seen_sha256[sha256] = record.image_id

            rows.append(
                {
                    "image_id": record.image_id,
                    "image_path": str(path),
                    "severity_label": record.severity_label,
                    "severity_name": self.severity_map.get(record.severity_label, "unknown"),
                    "image_exists": exists,
                    "file_size_bytes": size_bytes,
                    "sha256": sha256,
                    "format": fmt,
                    "is_duplicate": is_duplicate,
                    "annotation_source": adapter.format_name,
                }
            )

        df = pd.DataFrame(rows)

        duplicate_count = int(df["is_duplicate"].sum()) if not df.empty else 0
        if duplicate_count:
            self._logger.warning(
                "ManifestBuilder: %d duplicate image(s) detected (same SHA-256).",
                duplicate_count,
            )
        else:
            self._logger.info("ManifestBuilder: no duplicate images detected.")

        # Save manifest
        manifest_path = self.outputs_root / "manifest.csv"
        df.to_csv(manifest_path, index=False)
        self._logger.info(
            "ManifestBuilder: manifest saved → '%s' (%d rows).",
            manifest_path,
            len(df),
        )

        # Cross-reference all_folder
        if self._all_folder_name:
            self._cross_reference_all_folder(df)
        else:
            self._logger.info(
                "ManifestBuilder: 'ingestion.all_folder' not configured — "
                "skipping cross-reference."
            )

        # Ingestion log
        self._save_ingestion_log(df, adapter)

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cross_reference_all_folder(self, manifest_df: pd.DataFrame) -> None:
        """Compare ``all_folder`` contents against the manifest SHA-256 set.

        Classifies each file as:
          * ``in_class_folder``    — SHA-256 matches a manifest entry
          * ``unlabelled``         — SHA-256 not found in manifest
          * ``duplicate_in_all``   — SHA-256 seen more than once within all_folder

        Saves ``outputs_root/all_folder_index.csv``.

        Args:
            manifest_df: The freshly built manifest DataFrame.
        """
        all_folder_path = self.data_root / self._all_folder_name
        if not all_folder_path.is_dir():
            self._logger.warning(
                "ManifestBuilder: all_folder '%s' does not exist — skipping.",
                all_folder_path,
            )
            self._all_folder_summary = {
                "all_folder_path": str(all_folder_path),
                "error": "directory not found",
            }
            return

        # Build manifest SHA-256 → image_id lookup
        manifest_sha_to_id: dict[str, str] = {}
        if not manifest_df.empty and "sha256" in manifest_df.columns:
            for _, row in manifest_df.iterrows():
                if row["sha256"]:
                    manifest_sha_to_id[row["sha256"]] = row["image_id"]

        IMAGE_EXTENSIONS: frozenset[str] = frozenset(
            {".jpg", ".jpeg", ".png", ".webp"}
        )

        all_files = sorted(
            p
            for p in all_folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        # First pass: compute SHA-256 for all files
        sha_count: dict[str, int] = {}
        file_sha: dict[Path, str] = {}
        for p in all_files:
            try:
                h = compute_sha256(p)
            except OSError:
                h = ""
            file_sha[p] = h
            if h:
                sha_count[h] = sha_count.get(h, 0) + 1

        # Second pass: classify
        index_rows: list[dict] = []
        seen_in_all: set[str] = set()

        for p in all_files:
            h = file_sha[p]
            matched_id = manifest_sha_to_id.get(h, "")

            if not h:
                status = "unlabelled"
            elif sha_count.get(h, 1) > 1 and h in seen_in_all:
                status = "duplicate_in_all"
            elif matched_id:
                status = "in_class_folder"
            else:
                status = "unlabelled"

            if h:
                seen_in_all.add(h)

            index_rows.append(
                {
                    "sha256": h,
                    "filename": p.name,
                    "file_path": str(p.resolve()),
                    "status": status,
                    "matched_class_folder": matched_id if status == "in_class_folder" else "",
                }
            )

        all_df = pd.DataFrame(index_rows)
        all_index_path = self.outputs_root / "all_folder_index.csv"
        all_df.to_csv(all_index_path, index=False)

        total = len(index_rows)
        in_class = sum(1 for r in index_rows if r["status"] == "in_class_folder")
        unlabelled = sum(1 for r in index_rows if r["status"] == "unlabelled")
        dup_in_all = sum(1 for r in index_rows if r["status"] == "duplicate_in_all")

        self._logger.info(
            "ManifestBuilder [all_folder]: total=%d | in_class_folder=%d | "
            "unlabelled=%d | duplicate_in_all=%d. Index saved → '%s'.",
            total,
            in_class,
            unlabelled,
            dup_in_all,
            all_index_path,
        )

        self._all_folder_summary = {
            "all_folder_path": str(all_folder_path),
            "total": total,
            "in_class_folder": in_class,
            "unlabelled": unlabelled,
            "duplicate_in_all": dup_in_all,
        }

    def _save_ingestion_log(
        self,
        df: pd.DataFrame,
        adapter: AnnotationAdapter,
    ) -> None:
        """Serialise a JSON ingestion log with high-level statistics.

        Args:
            df:      The manifest DataFrame.
            adapter: The adapter used for this ingestion run.
        """
        per_class_counts: dict[str, int] = {}
        images_not_found = 0

        if not df.empty:
            if "severity_name" in df.columns:
                per_class_counts = (
                    df.groupby("severity_name").size().to_dict()
                )
            if "image_exists" in df.columns:
                images_not_found = int((~df["image_exists"]).sum())

        log: dict[str, Any] = {
            "format_detected": adapter.format_name,
            "total_records": len(df),
            "per_class_counts": per_class_counts,
            "duplicate_count": int(df["is_duplicate"].sum()) if not df.empty else 0,
            "images_not_found": images_not_found,
            "all_folder_discrepancy": self._all_folder_summary,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        log_path = self.outputs_root / "ingestion_log.json"
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=2, ensure_ascii=False)

        self._logger.info(
            "ManifestBuilder: ingestion log saved → '%s'.", log_path
        )
