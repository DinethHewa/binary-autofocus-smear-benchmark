from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def leakage_report(
    df: pd.DataFrame,
    group_col: str = "stack_id",
    split_col: str = "split",
    dataset_col: str = "dataset",
) -> Dict[str, List[str]]:
    leaks: Dict[str, List[str]] = {}

    if dataset_col in df.columns:
        datasets = df[dataset_col].dropna().unique()
        for dataset in datasets:
            subset = df[df[dataset_col] == dataset]
            groups_by_split: Dict[str, Set[str]] = {}
            for split in subset[split_col].unique():
                groups_by_split[split] = set(subset[subset[split_col] == split][group_col].unique())

            splits = list(groups_by_split.keys())
            for i, s1 in enumerate(splits):
                for s2 in splits[i + 1 :]:
                    overlap = groups_by_split[s1] & groups_by_split[s2]
                    if overlap:
                        key = f"{dataset}:{s1}__{s2}"
                        leaks[key] = sorted(overlap)
        return leaks

    groups_by_split = {
        split: set(df[df[split_col] == split][group_col].unique()) for split in df[split_col].unique()
    }
    splits = list(groups_by_split.keys())
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = groups_by_split[s1] & groups_by_split[s2]
            if overlap:
                key = f"{s1}__{s2}"
                leaks[key] = sorted(overlap)
    return leaks


def _sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_DCT_CACHE: Dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]
    x = np.arange(n)
    k = x[:, None]
    mat = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    mat = mat * math.sqrt(2.0 / n)
    mat[0, :] = mat[0, :] / math.sqrt(2.0)
    _DCT_CACHE[n] = mat
    return mat


def _phash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> Optional[str]:
    if Image is None:
        raise ImportError("PIL not installed; cannot compute perceptual hash.")
    img_size = hash_size * highfreq_factor
    with Image.open(path) as img:
        img = img.convert("L").resize((img_size, img_size), Image.BILINEAR)
        pixels = np.asarray(img, dtype=np.float32)
    dct_mat = _dct_matrix(img_size)
    dct = dct_mat @ pixels @ dct_mat.T
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])
    bits = dct_low > med
    hash_val = 0
    for bit in bits.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return f"{hash_val:0{hash_size * hash_size // 4}x}"


def _hash_leaks(
    df: pd.DataFrame,
    hash_fn,
    split_col: str,
    max_list: int,
) -> Tuple[int, List[str]]:
    seen: Dict[str, Dict[str, List[str]]] = {}
    total = len(df)
    hash_name = "sha1" if hash_fn is _sha1 else "phash" if hash_fn is _phash else "hash"
    if total >= 500:
        print(f"  {hash_name}: checking {total} files", flush=True)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        path = Path(row["image_path"])
        split = str(row[split_col])
        if not path.exists():
            raise FileNotFoundError(f"Missing image file: {path}")
        if hash_fn is _sha1 and "sha1" in row.index and pd.notna(row.get("sha1")):
            digest = str(row.get("sha1"))
        elif hash_fn is _phash and "phash" in row.index and pd.notna(row.get("phash")):
            digest = str(row.get("phash"))
        else:
            digest = hash_fn(path)
        if digest is None:
            continue
        entry = seen.setdefault(digest, {})
        entry.setdefault(split, []).append(str(path))
        if total >= 500 and (idx == 1 or idx % 250 == 0 or idx == total):
            print(f"  {hash_name}: checked {idx}/{total}", flush=True)

    offenders: List[str] = []
    for digest, split_map in seen.items():
        if len(split_map) > 1:
            for paths in split_map.values():
                offenders.extend(paths)

    offenders = sorted(set(offenders))
    return len(offenders), offenders[:max_list]


def audit_manifest(
    manifest_path: str | Path,
    *,
    group_col: str = "stack_id",
    split_col: str = "split",
    check_sha1: bool = True,
    check_phash: bool = True,
    max_list: int = 50,
) -> Dict[str, object]:
    manifest_path = Path(manifest_path)
    df = pd.read_csv(manifest_path)
    if split_col not in df.columns:
        raise KeyError(f"Manifest missing '{split_col}' column")

    report: Dict[str, object] = {
        "stack_leaks": {},
        "sha1": {"count": 0, "paths": []},
        "phash": {"count": 0, "paths": []},
    }

    stack_leaks = leakage_report(df, group_col=group_col, split_col=split_col)
    if stack_leaks:
        report["stack_leaks"] = stack_leaks

    if check_sha1:
        count, offenders = _hash_leaks(df, _sha1, split_col=split_col, max_list=max_list)
        report["sha1"] = {"count": count, "paths": offenders}

    if check_phash:
        if Image is None:
            raise ImportError("PIL not installed; leakage_phash requires pillow.")
        count, offenders = _hash_leaks(df, _phash, split_col=split_col, max_list=max_list)
        report["phash"] = {"count": count, "paths": offenders}

    _print_report(report, max_list=max_list)
    return report


def _print_report(report: Dict[str, object], max_list: int) -> None:
    stack_leaks = report.get("stack_leaks") or {}
    sha1 = report.get("sha1", {})
    phash = report.get("phash", {})

    print("Leakage audit summary:")
    if stack_leaks:
        total = sum(len(v) for v in stack_leaks.values())
        print(f"  stack_id overlaps: {total}")
        for key, items in list(stack_leaks.items())[:3]:
            print(f"    {key}: {len(items)}")
    else:
        print("  stack_id overlaps: 0")

    print(f"  sha1 duplicates: {sha1.get('count', 0)}")
    if sha1.get("paths"):
        for path in sha1["paths"][:max_list]:
            print(f"    {path}")

    print(f"  phash duplicates: {phash.get('count', 0)}")
    if phash.get("paths"):
        for path in phash["paths"][:max_list]:
            print(f"    {path}")


def assert_no_leakage(
    df: pd.DataFrame,
    group_col: str = "stack_id",
    split_col: str = "split",
) -> None:
    leaks = leakage_report(df, group_col=group_col, split_col=split_col)
    if leaks:
        total = sum(len(v) for v in leaks.values())
        raise ValueError(f"Leakage detected across splits: {total} overlapping groups.")


def assert_no_leakage_manifest(
    manifest_path: str | Path,
    *,
    group_col: str = "stack_id",
    split_col: str = "split",
    check_sha1: bool = True,
    check_phash: bool = True,
    max_list: int = 50,
) -> None:
    report = audit_manifest(
        manifest_path,
        group_col=group_col,
        split_col=split_col,
        check_sha1=check_sha1,
        check_phash=check_phash,
        max_list=max_list,
    )
    stack_leaks = report.get("stack_leaks") or {}
    sha1 = report.get("sha1", {})
    phash = report.get("phash", {})

    if stack_leaks or sha1.get("count", 0) or phash.get("count", 0):
        raise ValueError("Leakage detected across splits; see audit summary above.")
