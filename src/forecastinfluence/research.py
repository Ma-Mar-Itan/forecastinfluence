"""Matched finite-effect fidelity and anomaly-score alignment for research.

These statistics measure numerical approximation or descriptive association.
They are neither causal effects nor confidence intervals nor anomaly accuracy.
"""

from numbers import Real
from typing import Any, cast

import numpy as np
import pandas as pd

from .core import ForecastInfluenceError
from .diagnostics import compare
from .results import InfluenceResult


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    # Scaling preserves correlation and avoids squaring potentially huge units.
    return float(np.corrcoef(left / np.max(abs(left)), right / np.max(abs(right)))[0, 1])


def approximation_metrics(
    approximation: InfluenceResult,
    reference: InfluenceResult,
    *,
    top_k: int = 10,
    rank_by: str = "absolute",
    relative_floor: float = 1e-10,
) -> pd.DataFrame:
    """Summarize matched finite contrasts separately at every non-source coordinate.

    Reference must be a finite effect. A derivative must first be explicitly
    converted using ``first_order(change=...)``; raw derivatives cannot be
    compared directly with deletion effects. Compatibility checks include source
    units, membership, intervention magnitude, baseline, truth, replay and axes.

    Pearson uses signed effects. Spearman and top-k use absolute effects by
    default (or signed effects with rank_by='signed'). Spearman uses average
    ranks for ties. Top-k ties use stable source order and are flagged; k is
    capped to the number of jointly finite sources. Missing entries remain
    excluded and counted, never fabricated as zero. No origins/horizons/targets
    or parameter coordinates are silently pooled. Constant/insufficient vectors
    have undefined correlations reported as NaN.
    """
    if (
        approximation.metadata.effect_kind not in {"first_order_finite_effect", "finite_effect"}
        or reference.metadata.effect_kind != "finite_effect"
    ):
        raise ForecastInfluenceError(
            "Use first_order(change=...) or a finite estimate against a matched finite-effect reference."
        )
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ForecastInfluenceError("top_k must be a positive integer.")
    if rank_by not in {"absolute", "signed"}:
        raise ForecastInfluenceError("rank_by must be absolute or signed.")
    with np.errstate(over="ignore", invalid="ignore"):
        table = compare(approximation, reference, relative_floor=relative_floor)
    axes = list(approximation.dimensions[1:])
    rows = []
    for coordinate, group in table.groupby(axes, sort=False, dropna=False):
        coordinate = coordinate if isinstance(coordinate, tuple) else (coordinate,)
        finite = np.isfinite(group.estimate.to_numpy()) & np.isfinite(group.reference.to_numpy())
        valid = group.loc[finite]
        estimate = valid.estimate.to_numpy(dtype=float)
        truth = valid.reference.to_numpy(dtype=float)
        n = len(valid)
        ranked_estimate = abs(estimate) if rank_by == "absolute" else estimate
        ranked_truth = abs(truth) if rank_by == "absolute" else truth
        ranks_estimate = pd.Series(ranked_estimate).rank(method="average").to_numpy()
        ranks_truth = pd.Series(ranked_truth).rank(method="average").to_numpy()
        k = min(top_k, n)
        top_estimate = np.argsort(-ranked_estimate, kind="stable")[:k]
        top_truth = np.argsort(-ranked_truth, kind="stable")[:k]
        errors = valid.signed_error.to_numpy(dtype=float)
        if not np.isfinite(errors).all() or not np.isfinite(valid.relative_error.to_numpy()).all():
            raise ForecastInfluenceError(
                "Effect discrepancies overflowed; rescale units explicitly."
            )
        scale = float(np.max(abs(errors))) if n else 0.0
        scaled_errors = errors / scale if scale else errors
        ties = len(np.unique(ranked_estimate)) < n or len(np.unique(ranked_truth)) < n
        rows.append(
            {
                **dict(zip(axes, coordinate, strict=True)),
                "n_sources": len(group),
                "n_compared": n,
                "n_unavailable": len(group) - n,
                "mean_signed_error": float(np.mean(scaled_errors) * scale) if n else float("nan"),
                "mean_absolute_error": float(np.mean(abs(scaled_errors)) * scale)
                if n
                else float("nan"),
                "root_mean_squared_error": float(np.sqrt(np.mean(scaled_errors**2)) * scale)
                if n
                else float("nan"),
                "max_absolute_error": float(np.max(abs(errors))) if n else float("nan"),
                "mean_relative_error": float(valid.relative_error.mean()) if n else float("nan"),
                "pearson_signed": _correlation(estimate, truth),
                "spearman": _correlation(ranks_estimate, ranks_truth),
                "rank_by": rank_by,
                "rank_ties": ties,
                "top_k_requested": top_k,
                "top_k_used": k,
                "top_k_overlap": len(set(top_estimate) & set(top_truth)) / k if k else float("nan"),
                "sign_agreement": float(np.mean(np.sign(estimate) == np.sign(truth)))
                if n
                else float("nan"),
                "n_approximation_warnings": int(
                    (valid.estimate_status == "approximation_warning").sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.attrs = {
        "relative_floor": relative_floor,
        "meaning": "matched finite-effect numerical fidelity; not statistical uncertainty",
        "top_k_tie_policy": "stable source order",
        "reference_kind": reference.metadata.effect_kind,
    }
    return result


def anomaly_alignment(
    influence: InfluenceResult,
    anomaly_scores: pd.Series,
    *,
    influence_threshold: float,
    anomaly_threshold: float,
    **selectors: Any,
) -> pd.DataFrame:
    """Align user anomaly scores by explicit result source ID after axis selection.

    Supply scalar selectors for every non-singleton non-source axis, as for
    result.rank(). Thresholds are explicit: high influence means absolute effect
    >= influence_threshold, high anomaly means score >= anomaly_threshold.
    Scores must be indexed by source IDs, not ambiguous positional row offsets.
    Missing scores or effects are 'unavailable'; extra score IDs are rejected.
    Categories describe these two thresholds only. Low forecast influence does
    not prove harmlessness, and high influence does not prove an anomaly.
    """
    if not isinstance(anomaly_scores, pd.Series) or not anomaly_scores.index.is_unique:
        raise ForecastInfluenceError("anomaly_scores must be a Series with unique source IDs.")
    if np.iscomplexobj(anomaly_scores.to_numpy()):
        raise ForecastInfluenceError("Anomaly scores must be real.")
    try:
        scores = anomaly_scores.astype(float)
    except (TypeError, ValueError) as exc:
        raise ForecastInfluenceError("Anomaly scores must be numeric.") from exc
    if np.isinf(scores.to_numpy(dtype=float)).any():
        raise ForecastInfluenceError("Anomaly scores must be finite or NaN for unavailable.")
    if (
        any(
            not isinstance(v, Real) or not np.isfinite(float(v))
            for v in (influence_threshold, anomaly_threshold)
        )
        or influence_threshold < 0
    ):
        raise ForecastInfluenceError(
            "Thresholds must be finite real scalars; influence_threshold is nonnegative."
        )
    frame = influence.rank(**selectors)
    if not set(scores.index) <= set(frame.source):
        raise ForecastInfluenceError(
            "Anomaly scores contain unknown source IDs for this selection."
        )
    frame["anomaly_score"] = frame.source.map(scores)
    frame["high_influence"] = frame.absolute >= influence_threshold
    frame["high_anomaly"] = frame.anomaly_score >= anomaly_threshold
    categories = []
    for row in frame.itertuples():
        if not np.isfinite(float(cast(Any, row.effect))) or not np.isfinite(
            float(cast(Any, row.anomaly_score))
        ):
            categories.append("unavailable")
        elif row.high_influence and row.high_anomaly:
            categories.append("high anomaly, high influence")
        elif row.high_influence:
            categories.append("low anomaly, high influence")
        elif row.high_anomaly:
            categories.append("high anomaly, low influence")
        else:
            categories.append("low anomaly, low influence")
    frame["category"] = categories
    frame.attrs = {
        "influence_threshold": float(influence_threshold),
        "anomaly_threshold": float(anomaly_threshold),
        "meaning": "descriptive score alignment; neither anomaly ground truth nor causal harmfulness",
        "influence_units": influence.metadata.units,
        "effect_kind": influence.metadata.effect_kind,
    }
    return frame
