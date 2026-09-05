"""Explicit feature preprocessing and deterministic chronological grid replay."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .core import FloatArray, ForecastInfluenceError, ObjectiveSpec, ReplayPolicy
from .features import DesignMatrix
from .models import _immutable, _matrix


@dataclass(frozen=True)
class ScaleState:
    """Training-only feature location/scale. Responses remain in original units."""

    center: FloatArray
    scale: FloatArray
    method: str

    @classmethod
    def fit(cls, X: Any, method: str) -> "ScaleState":
        values = _matrix(X)
        magnitude = np.max(abs(values), axis=0) if len(values) else np.ones(values.shape[1])
        magnitude = np.where(magnitude == 0, 1.0, magnitude)
        normalized = values / magnitude
        if method == "identity":
            center, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])
        elif method == "standard":
            center, scale = normalized.mean(axis=0) * magnitude, normalized.std(axis=0) * magnitude
        elif method == "robust":
            center = np.median(normalized, axis=0) * magnitude
            scale = (
                np.quantile(normalized, 0.75, axis=0) - np.quantile(normalized, 0.25, axis=0)
            ) * magnitude
        else:
            raise ForecastInfluenceError("preprocessing must be identity, standard or robust.")
        # A constant predictor stays zero after centering. This is a scale convention,
        # not a solver rank rescue; OLS still rejects nonunique coefficients.
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ForecastInfluenceError(
                "Feature scaling overflowed; rescale original units explicitly."
            )
        scale = np.where(scale == 0, 1.0, scale)
        return cls(_immutable(center), _immutable(scale), method)

    def transform(self, X: Any) -> FloatArray:
        """Apply frozen training statistics to observed or forecast predictors."""
        return (_matrix(X) - self.center) / self.scale


@dataclass(frozen=True)
class ChronologicalGrid:
    """Expanding/rolling one-case validation, with train target <= validation issue.

    Candidate order resolves exact ties. No random splits or final-test outcomes.
    n_splits selects the last eligible chronological validation rows.
    """

    candidates: tuple[Any, ...]
    n_splits: int = 3
    min_train: int = 10
    train_window: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or any(not hasattr(c, "fit") for c in self.candidates):
            raise ForecastInfluenceError(
                "Supply a nonempty sequence of canonical regressor candidates."
            )
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (self.n_splits, self.min_train)
        ):
            raise ForecastInfluenceError("n_splits and min_train must be positive integers.")
        if self.train_window is not None and (
            not isinstance(self.train_window, int)
            or isinstance(self.train_window, bool)
            or self.train_window < self.min_train
        ):
            raise ForecastInfluenceError("train_window must be at least min_train.")


@dataclass(frozen=True)
class FittedPipeline:
    """Immutable original-feature parameter view and retained fitted procedure."""

    inner: Any
    scaler: ScaleState
    selected: Any
    selected_index: int
    scores: tuple[float, ...]
    folds: tuple[dict[str, Any], ...]
    switched: bool = False
    fit_count: int = 1

    @property
    def coefficients(self) -> FloatArray:
        """Slopes in original feature units, after undoing feature scaling."""
        return _immutable(self.inner.coefficients / self.scaler.scale)

    @property
    def support(self) -> np.ndarray:
        """Exact nonzero slope support, preserved under positive feature scaling."""
        return np.asarray(self.coefficients != 0)

    @property
    def intercept(self) -> float:
        """Intercept in original response units."""
        return float(self.inner.intercept - self.scaler.center @ self.coefficients)

    @property
    def parameters(self) -> FloatArray:
        """Original-unit intercept followed by slopes."""
        return _immutable(
            np.r_[self.intercept, self.coefficients]
            if self.objective.fit_intercept
            else self.coefficients
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Original feature labels."""
        return self.inner.parameter_names

    @property
    def objective(self) -> ObjectiveSpec:
        """Objective in the transformed-feature coefficient parameterization."""
        return self.inner.objective

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Solver, preprocessing, candidate scores and temporal fold provenance."""
        return {
            **self.inner.diagnostics,
            "preprocessing": self.scaler.method,
            "feature_center": self.scaler.center.tolist(),
            "feature_scale": self.scaler.scale.tolist(),
            "penalty_parameterization": "transformed_feature_coefficients",
            "selected_candidate": self.selected_index,
            "selected_spec": repr(self.selected),
            "validation_scores": list(self.scores),
            "folds": [dict(f) for f in self.folds],
            "penalty_switched": self.switched,
            "fit_count": self.fit_count,
        }

    def predict(self, X: FloatArray) -> FloatArray:
        """Predict using the retained transformed-feature model."""
        return self.inner.predict(self.scaler.transform(X))


@dataclass(frozen=True)
class PipelineRegressor:
    """Canonical regressor with explicitly replayable feature scaling and tuning.

    Feature statistics are unweighted empirical statistics of retained rows.
    Reweighting thus changes fitting and validation weights, not the raw feature
    distribution. Raw edits change statistics when preprocessing='refit'.
    Tuning always fits fold-local preprocessing to avoid validation leakage.
    """

    regressor: Any
    preprocessing: str = "identity"
    tuning: ChronologicalGrid | None = None
    capabilities: ClassVar[frozenset[str]] = frozenset({"refit", "central_difference"})

    def __post_init__(self) -> None:
        if self.preprocessing not in {"identity", "standard", "robust"}:
            raise ForecastInfluenceError("Unknown preprocessing method.")
        candidates = self.tuning.candidates if self.tuning else (self.regressor,)
        if self.preprocessing != "identity" and any(not c.fit_intercept for c in candidates):
            raise ForecastInfluenceError(
                "Centered preprocessing requires an unpenalized fitted intercept."
            )
        if any(c.fit_intercept != self.regressor.fit_intercept for c in candidates):
            raise ForecastInfluenceError("All candidates must share the intercept convention.")

    @property
    def fit_intercept(self) -> bool:
        """Common intercept convention."""
        return bool(self.regressor.fit_intercept)

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        weights: FloatArray | None = None,
        n0: int | None = None,
        feature_names: tuple[str, ...] | None = None,
    ) -> FittedPipeline:
        """Fit scaling without tuning; tuning needs timestamped fit_design."""
        if self.tuning:
            raise ForecastInfluenceError(
                "Chronological tuning requires a timestamped DesignMatrix."
            )
        scaler = ScaleState.fit(X, self.preprocessing)
        model = self.regressor.fit(
            scaler.transform(X), y, weights=weights, n0=n0, feature_names=feature_names
        )
        return FittedPipeline(model, scaler, self.regressor, 0, (), ())

    def _select(
        self, design: DesignMatrix, weights: FloatArray
    ) -> tuple[Any, int, tuple[float, ...], tuple[dict[str, Any], ...]]:
        grid = self.tuning
        if grid is None:
            return self.regressor, 0, (), ()
        splits = []
        for row, issue in enumerate(design.issue_times):
            train = np.array(
                [i for i, target in enumerate(design.target_times) if i < row and target <= issue],
                dtype=int,
            )
            if grid.train_window:
                train = train[-grid.train_window :]
            if len(train) >= grid.min_train and weights[row] > 0 and weights[train].sum() > 0:
                splits.append((train, row))
        splits = splits[-grid.n_splits :]
        if len(splits) < grid.n_splits:
            raise ForecastInfluenceError("Insufficient eligible chronological validation folds.")
        scores = []
        for candidate in grid.candidates:
            loss, total = 0.0, 0.0
            for train, row in splits:
                scaler = ScaleState.fit(design.X[train], self.preprocessing)
                model = candidate.fit(
                    scaler.transform(design.X[train]),
                    design.y[train],
                    weights=weights[train],
                    n0=len(train),
                    feature_names=design.feature_names,
                )
                error = float(model.predict(scaler.transform(design.X[[row]]))[0] - design.y[row])
                loss += weights[row] * error**2
                total += weights[row]
            scores.append(float(loss / total))
        if not np.isfinite(scores).all():
            raise ForecastInfluenceError("Validation scores are nonfinite.")
        selected = int(np.argmin(scores))
        folds = tuple(
            {
                "validation_issue": str(design.issue_times[row]),
                "validation_target": str(design.target_times[row]),
                "latest_training_target": str(design.target_times[train[-1]]),
                "training_cases": len(train),
            }
            for train, row in splits
        )
        return grid.candidates[selected], selected, tuple(scores), folds

    def fit_design(
        self, design: DesignMatrix, *, weights: FloatArray | None = None
    ) -> FittedPipeline:
        """Fit a baseline procedure with chronological validation provenance."""
        return self.replay_design(
            design,
            None,
            weights=weights,
            policy=ReplayPolicy(preprocessing="refit", hyperparameters="retune"),
        )

    def replay_design(
        self,
        design: DesignMatrix,
        baseline: FittedPipeline | None,
        *,
        weights: FloatArray | None = None,
        policy: ReplayPolicy,
    ) -> FittedPipeline:
        """Replay declared fitted state, keeping final-fit n0 fixed."""
        if weights is not None and np.iscomplexobj(weights):
            raise ForecastInfluenceError("Pipeline weights must be real.")
        w = np.ones(len(design.y)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != design.y.shape or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ForecastInfluenceError("Invalid pipeline weights.")
        if (
            baseline is not None
            and self.tuning
            and len(design.y) != design.n0
            and policy.hyperparameters == "retune"
        ):
            raise ForecastInfluenceError(
                "Retuning after physical deletion needs frozen fold identities; use zero case weights or fixed tuning."
            )
        if baseline is None or policy.hyperparameters == "retune":
            selected, index, scores, folds = self._select(design, w)
        else:
            selected, index, scores, folds = (
                baseline.selected,
                baseline.selected_index,
                baseline.scores,
                baseline.folds,
            )
        scaler = (
            ScaleState.fit(design.X, self.preprocessing)
            if baseline is None or policy.preprocessing == "refit"
            else baseline.scaler
        )
        model = selected.fit(
            scaler.transform(design.X),
            design.y,
            weights=w,
            n0=design.n0,
            feature_names=design.feature_names,
        )
        return FittedPipeline(
            model,
            scaler,
            selected,
            index,
            scores,
            folds,
            baseline is not None and index != baseline.selected_index,
            1
            + (
                len(self.tuning.candidates) * self.tuning.n_splits
                if self.tuning and (baseline is None or policy.hyperparameters == "retune")
                else 0
            ),
        )
