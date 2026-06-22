from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision_utils import build_metadata, config_hash, ensure_dir, fresh_all, load_config, safe_write_text, save_metadata_for_outputs  # noqa: E402


SCRIPT_NAME = Path(__file__).name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LaTeX-ready metric equations.")
    parser.add_argument("--config", default="configs/bspc_revision.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


TEX = r"""\begin{align}
\mathrm{Sensitivity} &= \frac{TP}{TP + FN},\\
\mathrm{Specificity} &= \frac{TN}{TN + FP},\\
\mathrm{Balanced\ Accuracy} &= \frac{1}{2}\left(\mathrm{Sensitivity} + \mathrm{Specificity}\right),\\
\mathrm{Precision} &= \frac{TP}{TP + FP},\\
F_1 &= \frac{2\,\mathrm{Precision}\,\mathrm{Sensitivity}}{\mathrm{Precision}+\mathrm{Sensitivity}},\\
\mathrm{MCC} &= \frac{TP\,TN - FP\,FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}},\\
\mathrm{AUC} &= \int_0^1 \mathrm{TPR}(\mathrm{FPR})\,d\mathrm{FPR},\\
\mathrm{Brier} &= \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2,\\
\mathrm{ECE} &= \sum_{b=1}^{B}\frac{|I_b|}{N}\left|\mathrm{acc}(I_b)-\mathrm{conf}(I_b)\right|,\\
\mathrm{FAR} &= \frac{FP}{FP+TN},\\
\mathrm{FRR} &= \frac{FN}{FN+TP}.
\end{align}

For validation threshold candidates $t \in \mathcal{T}$, Calibration-Aware Operating-Point Selection (CAOPS) is defined as:

\begin{align}
t_{\lambda}^{*} &= \arg\max_{t \in \mathcal{T}}\left[\mathrm{BA}_{val}(t) - \lambda\,\mathrm{ECE}_{val}(t)\right],\\
t_{\delta}^{*} &= \arg\max_{t \in \mathcal{T}: \mathrm{ECE}_{val}(t) \leq \delta}\mathrm{BA}_{val}(t).
\end{align}
"""


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg_hash = config_hash(config)
    out_dir = ensure_dir((config.get("subdirs") or {}).get("paper_exports", "revision_outputs/paper_exports"))
    output = out_dir / "metric_equations.tex"
    if not args.force and fresh_all([output], [args.config], cfg_hash):
        print("metric equation export is fresh; skipping")
        return 0
    safe_write_text(TEX, output)
    metadata = build_metadata(script_name=SCRIPT_NAME, input_files=[args.config], cfg_hash=cfg_hash, args=args)
    save_metadata_for_outputs([output], metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
