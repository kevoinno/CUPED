"""Export each frame of the CUPED Visualized animation to images/.

The notebook cycles through three frames on a timer, so the steps are never all
on screen at once. This renders each one to a PNG for use in the README, slides
or anywhere else static images are needed.

The charts are not rebuilt here. `app.run()` executes the notebook and hands
back its definitions, including `viz_build_chart`, so these images are produced
by the very same code path that draws the live figure and cannot drift from it.

Usage:
    uv run python scripts/export_steps.py [--scale 2] [--out images]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "images",
        help="directory to write the PNGs into (default: images/)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="pixel scale factor; 2 gives a retina-sharp image (default: 2)",
    )
    args = parser.parse_args()

    from CUPED_simulation import app

    print("running the notebook ...")
    _outputs, defs = app.run()

    build = defs["viz_build_chart"]
    labels = defs["viz_stage_labels"]

    args.out.mkdir(parents=True, exist_ok=True)

    for step, label in enumerate(labels):
        chart = build(step)
        path = args.out / f"cuped_step_{step}.png"
        # Needs vl-convert-python, declared in the project's dev dependencies.
        chart.save(path, scale_factor=args.scale)
        print(f"  step {step}  {label:<22} -> {path.relative_to(REPO_ROOT)}")

    print(f"\nwrote {len(labels)} images to {args.out.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
