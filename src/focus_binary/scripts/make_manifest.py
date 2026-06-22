from __future__ import annotations

import argparse
import re
from pathlib import Path

from focus_binary import paths
from focus_binary.data.discover import scan_datasets
from focus_binary.data.manifest import Manifest, default_manifest_path
from focus_binary.data.splits import assert_no_leak, split_manifest, write_split_meta
from focus_binary.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified manifest across datasets.")
    parser.add_argument("--root", required=True, help="Root folder containing dataset subfolders")
    parser.add_argument("--out", default=None, help="Destination CSV path for manifest")
    parser.add_argument(
        "--stack-regex",
        default=None,
        help="Optional regex to extract stack_id from filename if no stack folder exists (use capture group 1).",
    )
    parser.add_argument(
        "--extensions",
        default=".png,.jpg,.jpeg,.tif,.tiff",
        help="Comma-separated list of allowed image extensions.",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=None, help="Optional cap for discovery (debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split assignment")
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--group-col", default="stack_id", help="Group column for leakage-free splits")
    parser.add_argument("--stratify-col", default="label", help="Column to stratify at group level")
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        default=True,
        help="Split each dataset independently (default: true)",
    )
    parser.add_argument(
        "--no-by-dataset",
        dest="by_dataset",
        action="store_false",
        help="Split across all datasets jointly",
    )
    parser.add_argument("--split-out", default=None, help="Destination CSV for manifest with splits")
    parser.add_argument("--meta-out", default=None, help="Destination JSON for split metadata")
    return parser.parse_args(argv)


def _print_summary(manifest: Manifest) -> None:
    df = manifest.df
    print(f"Total images: {len(df)}")

    print("Images per dataset:")
    for dataset, count in df.groupby("dataset").size().items():
        print(f"  {dataset}: {count}")

    print("Images per dataset/class (label 1=focused, 0=unfocused):")
    for (dataset, label), count in df.groupby(["dataset", "label"]).size().items():
        print(f"  {dataset} label={label}: {count}")

    print("Unique stacks per dataset/class:")
    stack_counts = df.groupby(["dataset", "label"])["stack_id"].nunique()
    for (dataset, label), count in stack_counts.items():
        print(f"  {dataset} label={label}: {count} stacks")


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)

    output_root = Path(args.root).expanduser().resolve()
    exts = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    stack_pattern = re.compile(args.stack_regex) if args.stack_regex else None

    scans = scan_datasets(
        output_root=output_root,
        image_exts=exts,
        stack_regex=stack_pattern,
        source=output_root.name,
        limit_per_dataset=args.limit_per_dataset,
    )

    manifest = Manifest.from_scans(scans).validate()

    dest = Path(args.out) if args.out else default_manifest_path()
    if not dest.is_absolute():
        dest = paths.PROJECT_ROOT / dest

    manifest.to_csv(dest)
    split_df = split_manifest(
        manifest.df,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        by_dataset=args.by_dataset,
    )

    split_dest = Path(args.split_out) if args.split_out else dest.with_name(f"{dest.stem}_with_splits.csv")
    if not split_dest.is_absolute():
        split_dest = paths.PROJECT_ROOT / split_dest
    split_dest.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(split_dest, index=False)

    meta_dest = Path(args.meta_out) if args.meta_out else split_dest.with_name("splits_meta.json")
    if not meta_dest.is_absolute():
        meta_dest = paths.PROJECT_ROOT / meta_dest

    write_split_meta(
        meta_dest,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
        group_col=args.group_col,
        stratify_col=args.stratify_col,
        by_dataset=args.by_dataset,
    )

    assert_no_leak(split_df, group_col=args.group_col, split_col="split")
    _print_summary(manifest)
    return dest


if __name__ == "__main__":
    main()
