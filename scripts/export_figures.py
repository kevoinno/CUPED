"""Export the notebook's figures to images/.

Two sets:

* the three frames of the CUPED Visualized animation, which cycle on a timer
  and so are never all on screen at once;
* the 95% confidence interval for each estimator, one image per method, for
  placing side by side in a README or slide.

`app.run()` executes the notebook and hands back its definitions, so every
number here is the notebook's own. The animation frames additionally reuse the
notebook's `viz_build_chart`, so they cannot drift from the live figure. The
interval charts are laid out here, because the notebook draws both estimators
in a single combined chart and these are one-per-image.

Both interval charts are pinned to the same y-axis. Without that the narrower
interval would not read as narrower, which is the only thing the pair is for.

Usage:
    uv run python scripts/export_figures.py [--scale 2] [--out images]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NAIVE_COLOR = "#6366f1"
CUPED_COLOR = "#10b981"
INK = "#1e293b"


def export_steps(defs, out: Path, scale: float) -> list[Path]:
    """The three animation frames, straight from the notebook's own builder."""
    build = defs["viz_build_chart"]
    written = []
    for step, label in enumerate(defs["viz_stage_labels"]):
        path = out / f"cuped_step_{step}.png"
        build(step).save(path, scale_factor=scale)
        print(f"  step {step}  {label:<22} -> {path.name}")
        written.append(path)
    return written


def _ci_chart(alt, pd, *, label, effect, lo, hi, color, tau, domain):
    frame = pd.DataFrame({"method": [label], "effect": [effect], "lo": [lo], "hi": [hi]})
    y = alt.Y(
        "lo:Q",
        scale=alt.Scale(domain=domain, nice=False),
        title="Treatment effect estimate",
    )
    x = alt.X("method:N", title=None, axis=alt.Axis(labelFontSize=13, labelAngle=0))

    true_effect = (
        alt.Chart(pd.DataFrame({"y": [tau]}))
        .mark_rule(color=INK, strokeDash=[6, 4], strokeWidth=2)
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=domain, nice=False)))
    )
    interval = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=7, strokeCap="round", color=color)
        .encode(x=x, y=y, y2="hi:Q")
    )
    point = (
        alt.Chart(frame)
        .mark_point(size=190, filled=True, color=color, opacity=1, stroke="white", strokeWidth=2)
        .encode(x=x, y=alt.Y("effect:Q", scale=alt.Scale(domain=domain, nice=False)))
    )
    bounds = (
        alt.Chart(frame)
        .transform_fold(["lo", "hi"], as_=["which", "v"])
        .mark_text(dx=26, fontSize=11, color="#475569")
        .encode(
            x=x,
            y=alt.Y("v:Q", scale=alt.Scale(domain=domain, nice=False)),
            text=alt.Text("v:Q", format=".2f"),
        )
    )
    estimate = (
        alt.Chart(frame)
        .mark_text(dx=26, fontSize=12, fontWeight="bold", color=INK)
        .encode(
            x=x,
            y=alt.Y("effect:Q", scale=alt.Scale(domain=domain, nice=False)),
            text=alt.Text("effect:Q", format=".2f"),
        )
    )
    return alt.layer(true_effect, interval, point, bounds, estimate).properties(
        # Wide enough that the title text, not the plot, stops driving the
        # canvas width -- otherwise the two images come out different sizes
        # and will not sit neatly side by side.
        width=300,
        height=430,
        title=alt.TitleParams(
            text=f"{label} — 95% confidence interval",
            subtitle=[
                f"width {hi - lo:.2f}  ·  dashed line = true effect {tau:.1f}",
            ],
            fontSize=14,
            subtitleFontSize=11,
            anchor="start",
        ),
    )


def export_ci_charts(defs, out: Path, scale: float) -> list[Path]:
    """One image per estimator, pinned to a shared y-axis so they compare."""
    import altair as alt
    import pandas as pd

    tau = float(defs["tau"])
    lo = min(defs["naive_ci_low"], defs["cuped_ci_low"], tau)
    hi = max(defs["naive_ci_high"], defs["cuped_ci_high"], tau)
    pad = (hi - lo) * 0.12
    domain = [float(lo - pad), float(hi + pad)]

    specs = [
        ("ci_standard_ab_test", "Standard A/B test", "naive", NAIVE_COLOR),
        ("ci_cuped", "CUPED", "cuped", CUPED_COLOR),
    ]
    written = []
    for filename, label, prefix, color in specs:
        chart = _ci_chart(
            alt,
            pd,
            label=label,
            effect=float(defs[f"{prefix}_effect"]),
            lo=float(defs[f"{prefix}_ci_low"]),
            hi=float(defs[f"{prefix}_ci_high"]),
            color=color,
            tau=tau,
            domain=domain,
        )
        path = out / f"{filename}.png"
        chart.save(path, scale_factor=scale)
        width = defs[f"{prefix}_ci_high"] - defs[f"{prefix}_ci_low"]
        print(f"  {label:<20} CI width {width:6.2f} -> {path.name}")
        written.append(path)
    return written


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

    args.out.mkdir(parents=True, exist_ok=True)

    # Needs vl-convert-python, declared in the project's dev dependencies.
    written = export_steps(defs, args.out, args.scale)
    written += export_ci_charts(defs, args.out, args.scale)

    print(f"\nwrote {len(written)} images to {args.out.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
