from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from focus_binary.data.discover import DatasetScan, Sample, flatten_scans
from focus_binary.utils.logging import get_logger
from focus_binary import paths

logger = get_logger(__name__)


COLUMNS = ["dataset", "image_path", "label", "stack_id", "patient_id", "source"]


@dataclass
class Manifest:
    df: pd.DataFrame

    @classmethod
    def empty(cls) -> "Manifest":
        return cls(pd.DataFrame(columns=COLUMNS))

    @classmethod
    def from_samples(cls, samples: Iterable[Sample]) -> "Manifest":
        records = [
            {
                "dataset": sample.dataset,
                "image_path": str(sample.path),
                "label": sample.label,
                "stack_id": sample.stack_id,
                "patient_id": sample.patient_id,
                "source": sample.source,
            }
            for sample in samples
        ]
        df = pd.DataFrame.from_records(records, columns=COLUMNS)
        return cls(df)

    @classmethod
    def from_scans(cls, scans: Iterable[DatasetScan]) -> "Manifest":
        return cls.from_samples(flatten_scans(list(scans)))

    def validate(self) -> "Manifest":
        missing = self.df[~self.df["image_path"].apply(lambda p: Path(p).exists())]
        if not missing.empty:
            raise FileNotFoundError(f"Missing files referenced in manifest: {len(missing)}")

        labels = set(self.df["label"].unique())
        if not labels.issubset({0, 1}):
            raise ValueError(f"Invalid labels found in manifest: {labels}")
        return self

    def to_csv(self, path: Path, index: bool = False) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=index)
        logger.info("wrote manifest", extra={"rows": len(self.df), "path": str(path)})
        return path

    def summary_counts(self) -> pd.DataFrame:
        return self.df.groupby(["dataset", "label"]).size().rename("count").reset_index()

    def stack_counts(self) -> pd.DataFrame:
        return self.df.groupby(["dataset", "label"])["stack_id"].nunique().rename("unique_stacks").reset_index()


# Convenience helpers

def default_manifest_path() -> Path:
    return paths.ARTIFACT_DIR / "manifest.csv"
