"""Canonical fixed-denominator LASSO and elastic-net solver adapters."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import import_module
from numbers import Integral, Real
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray, NumericalError, ObjectiveSpec, UnsupportedCapabilityError
from .models import RidgeRegressor, _immutable, _matrix


def _nonnegative(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}.")
    return result


def _dependency(name: str) -> ModuleType:
    try:
        return import_module(name)
    except ImportError as exc:
        raise UnsupportedCapabilityError(
            "This estimator requires optional numerical solvers; install forecastinfluence[models] or -e '.[models]' from a checkout."
        ) from exc


def _options(fit_intercept: bool, tolerance: float, max_iter: int) -> None:
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be a bool.")
    _nonnegative(tolerance, "tolerance", positive=True)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")


def _prepare(
    X: ArrayLike,
    y: ArrayLike,
    weights: ArrayLike | None,
    n0: int | None,
    feature_names: Sequence[str] | None,
    fit_intercept: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, int, tuple[str, ...]]:
    design = _matrix(X)
    if np.iscomplexobj(y) or (weights is not None and np.iscomplexobj(weights)):
        raise ValueError("Responses and weights must be real, not complex.")
    response = np.array(y, dtype=float, copy=True)
    n, p = design.shape
    if not n or response.shape != (n,) or not np.isfinite(response).all():
        raise ValueError("y must be a finite vector matching nonempty X rows.")
    count = n if n0 is None else n0
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1:
        raise ValueError("n0 must be a positive integer baseline denominator.")
    w = np.ones(n) if weights is None else np.array(weights, dtype=float, copy=True)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite, nonnegative and include a positive weight.")
    with np.errstate(over="ignore"):
        total = float(w.sum())
    if not np.isfinite(total):
        raise NumericalError("Total case weight overflowed; choose representable weight units.")
    names = tuple(f"x{i}" for i in range(p)) if feature_names is None else tuple(feature_names)
    if (
        len(names) != p
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != p
        or (fit_intercept and "intercept" in names)
    ):
        raise ValueError(
            "feature_names must be unique nonempty names excluding reserved intercept."
        )
    if not p and not fit_intercept:
        raise ValueError("A model requires features or an intercept.")
    return design, response, w, int(count), names


@dataclass(frozen=True, eq=False)
class FittedSparseModel:
    """Immutable sparse coefficients, exact solver support and KKT diagnostics.

    Attributes
    ----------
    parameters : ndarray of shape (parameter,)
        Optional intercept first, followed by slopes. Immutable float64 storage.
    objective : ObjectiveSpec
        Canonical L1/L2 penalties and the frozen baseline denominator.
    support : tuple of str
        Feature names with exactly nonzero fitted coefficients; intercept excluded.
    signs : ndarray of shape (feature,)
        Actual coefficient signs (-1, 0, 1), with no thresholding.
    diagnostics : dict
        Defensive fit diagnostics including convergence, KKT, support margins,
        uniqueness certificate, solver mapping, versions and tolerances.
    """

    parameters: FloatArray
    parameter_names: tuple[str, ...]
    objective: ObjectiveSpec
    objective_value: float
    _residuals: FloatArray = field(repr=False)
    _diagnostic_items: tuple[tuple[str, Any], ...] = field(repr=False)

    @property
    def coefficients(self) -> FloatArray:
        """Return immutable slopes in feature order."""
        return self.parameters[int(self.objective.fit_intercept) :]

    @property
    def intercept(self) -> float:
        """Return the unpenalized intercept, or zero when disabled."""
        return float(self.parameters[0]) if self.objective.fit_intercept else 0.0

    @property
    def support(self) -> tuple[str, ...]:
        """Return names of exactly nonzero slopes; no support tolerance is applied."""
        names = self.parameter_names[int(self.objective.fit_intercept) :]
        return tuple(
            name for name, value in zip(names, self.coefficients, strict=True) if value != 0
        )

    @property
    def active_set(self) -> tuple[str, ...]:
        """Alias for the exactly nonzero feature support."""
        return self.support

    @property
    def signs(self) -> FloatArray:
        """Return immutable actual coefficient signs, including zeros."""
        return _immutable(np.sign(self.coefficients))

    @property
    def residuals(self) -> FloatArray:
        """Return immutable response-minus-fitted-value residuals."""
        return self._residuals

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return a fresh dictionary whose stored values are immutable."""
        return dict(self._diagnostic_items)

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict finite responses from X of shape (case, feature)."""
        design = _matrix(X)
        if design.shape[1] != self.coefficients.size:
            raise ValueError("Prediction feature count differs from the sparse fit.")
        with np.errstate(over="ignore", invalid="ignore"):
            result = design @ self.coefficients + self.intercept
        if not np.isfinite(result).all():
            raise NumericalError("Sparse prediction is nonfinite; rescale inputs explicitly.")
        return result


def _fit_sparse(
    X: ArrayLike,
    y: ArrayLike,
    *,
    weights: ArrayLike | None,
    n0: int | None,
    feature_names: Sequence[str] | None,
    fit_intercept: bool,
    l1_penalty: float,
    l2_penalty: float,
    tolerance: float,
    max_iter: int,
) -> FittedSparseModel:
    design, response, w, count, names = _prepare(X, y, weights, n0, feature_names, fit_intercept)
    total = float(w.sum())
    p = design.shape[1]
    diagnostics: dict[str, Any] = {
        "numpy_version": np.__version__,
        "tolerance": tolerance,
        "max_iter": max_iter,
        "weight_sum": total,
        "n0": count,
        "standardization": "none",
        "selection": "cyclic",
        "warm_start": False,
    }
    if l1_penalty == 0 or p == 0:
        native = RidgeRegressor(l2_penalty, fit_intercept).fit(
            design, response, weights=w, n0=count, feature_names=names
        )
        parameters = native.parameters
        diagnostics.update(
            solver="native_augmented_lstsq",
            n_iter=0,
            dual_gap=0.0,
            external_alpha=0.0,
            external_l1_ratio=0.0,
        )
    else:
        sklearn = _dependency("sklearn")
        linear = _dependency("sklearn.linear_model")
        external_alpha = count * (l1_penalty + l2_penalty) / total
        if not np.isfinite(external_alpha) or external_alpha <= 0:
            raise NumericalError("Canonical-to-solver penalty mapping is not representable.")
        solver = linear.ElasticNet(
            alpha=external_alpha,
            l1_ratio=l1_penalty / (l1_penalty + l2_penalty),
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tolerance,
            selection="cyclic",
            warm_start=False,
            copy_X=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver.fit(design, response, sample_weight=w)
        if caught:
            messages = "; ".join(str(warning.message) for warning in caught)
            raise NumericalError(
                f"Sparse solver emitted a warning: {messages}. Increase max_iter or rescale explicitly."
            )
        parameters = (
            np.r_[solver.intercept_, solver.coef_] if fit_intercept else np.asarray(solver.coef_)
        )
        diagnostics.update(
            solver="sklearn.ElasticNet_coordinate_descent",
            sklearn_version=sklearn.__version__,
            n_iter=int(solver.n_iter_),
            dual_gap=float(solver.dual_gap_),
            external_alpha=external_alpha,
            external_l1_ratio=solver.l1_ratio,
        )
    slopes = parameters[int(fit_intercept) :]
    with np.errstate(over="ignore", invalid="ignore"):
        residual = response - design @ slopes - (parameters[0] if fit_intercept else 0.0)
        gradient = -(design.T @ (w * residual)) / count + l2_penalty * slopes
        active = slopes != 0
        violation = np.where(
            active,
            np.abs(gradient + l1_penalty * np.sign(slopes)),
            np.maximum(np.abs(gradient) - l1_penalty, 0),
        )
        intercept_violation = abs(float(w @ residual / count)) if fit_intercept else 0.0
        kkt_error = max(float(violation.max(initial=0)), intercept_violation)
        gradient_scale = max(1.0, float(np.abs(design.T @ (w * response) / count).max(initial=0)))
        kkt_tolerance = max(1e-10, 20 * tolerance * gradient_scale)
        loss = float(w @ residual**2 / (2 * count))
        penalty = float(l1_penalty * np.abs(slopes).sum() + l2_penalty * (slopes @ slopes) / 2)
    if not np.isfinite(parameters).all() or not np.isfinite(
        loss + penalty + kkt_error + gradient_scale + kkt_tolerance
    ):
        raise NumericalError("Sparse fit or objective diagnostics are nonfinite.")
    if kkt_error > kkt_tolerance:
        raise NumericalError(
            f"Sparse KKT residual {kkt_error:g} exceeds declared check {kkt_tolerance:g}; reduce tolerance or rescale."
        )
    # Full rank of the equicorrelation subproblem is a sufficient uniqueness
    # certificate for LASSO. Failure is conservative, never hidden damping.
    slack = l1_penalty - np.abs(gradient)
    equicorrelation = active | (slack <= kkt_tolerance)
    reduced = design[:, equicorrelation]
    if fit_intercept:
        reduced = np.column_stack([np.ones(len(response)), reduced])
    augmented = reduced * np.sqrt(w)[:, None]
    reduced_p = reduced.shape[1]
    if l2_penalty and reduced_p:
        diagonal = np.ones(reduced_p)
        if fit_intercept:
            diagonal[0] = 0
        augmented = np.vstack([augmented, np.diag(np.sqrt(count * l2_penalty) * diagonal)])
    rank = int(np.linalg.matrix_rank(augmented)) if reduced_p else 0
    if rank != reduced_p:
        raise NumericalError(
            "Sparse optimum uniqueness could not be certified: rank-deficient equicorrelation set. Use explicit positive L2 or revise features."
        )
    diagnostics.update(
        converged=True,
        kkt_residual=kkt_error,
        kkt_tolerance=kkt_tolerance,
        active_set_size=int(active.sum()),
        support_threshold=0.0,
        min_active_magnitude=float(np.abs(slopes[active]).min()) if active.any() else None,
        inactive_kkt_slack_min=float(slack[~active].min()) if (~active).any() else None,
        equicorrelation_rank=rank,
        equicorrelation_size=reduced_p,
        uniqueness="full_rank_equicorrelation_subproblem",
        objective_loss=loss,
        objective_penalty=penalty,
        objective_value=loss + penalty,
        implicit_supported=False,
        support_path="numerical_refits_only",
    )
    return FittedSparseModel(
        _immutable(parameters),
        (("intercept",) if fit_intercept else ()) + names,
        ObjectiveSpec(count, l2_penalty, fit_intercept, l1_penalty=l1_penalty),
        loss + penalty,
        _immutable(residual),
        tuple(diagnostics.items()),
    )


@dataclass(frozen=True)
class LassoRegressor:
    """LASSO with canonical L1 penalty and frozen baseline normalization.

    Parameters
    ----------
    penalty : float, default 0.1
        Lambda1 multiplying the absolute slope sum; intercept unpenalized.
    fit_intercept : bool, default True
        Fit an unpenalized intercept.
    tolerance : float, default 1e-8
        External coordinate-descent stopping tolerance; KKT checks also apply.
    max_iter : int, default 100000
        Maximum deterministic cyclic coordinate-descent iterations.

    Notes
    -----
    Numerical refits and central differences are supported. No smooth implicit
    derivative is advertised at active-set transitions or within active regions.
    """

    penalty: float = 0.1
    fit_intercept: bool = True
    tolerance: float = 1e-8
    max_iter: int = 100_000
    capabilities: ClassVar[frozenset[str]] = frozenset({"refit", "central_difference"})

    def __post_init__(self) -> None:
        _nonnegative(self.penalty, "penalty")
        _options(self.fit_intercept, self.tolerance, self.max_iter)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        n0: int | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> FittedSparseModel:
        """Fit finite X(case, feature), y(case), absolute weights and fixed n0.

        Returns an immutable snapshot; feature_names label the predictor columns.
        Nonconvergence or an uncertified unique solution raises NumericalError.
        """
        return _fit_sparse(
            X,
            y,
            weights=weights,
            n0=n0,
            feature_names=feature_names,
            fit_intercept=self.fit_intercept,
            l1_penalty=self.penalty,
            l2_penalty=0.0,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
        )


@dataclass(frozen=True)
class ElasticNetRegressor:
    """Elastic net with separate canonical L1 and L2 penalties.

    Parameters
    ----------
    l1_penalty : float, default 0.1
        Lambda1 multiplying the absolute slope sum.
    l2_penalty : float, default 0.1
        Lambda2 multiplying half the squared slope norm.
    fit_intercept : bool, default True
        Include an unpenalized intercept.
    tolerance : float, default 1e-8
        Deterministic coordinate-descent stopping tolerance.
    max_iter : int, default 100000
        Maximum coordinate-descent iterations.
    """

    l1_penalty: float = 0.1
    l2_penalty: float = 0.1
    fit_intercept: bool = True
    tolerance: float = 1e-8
    max_iter: int = 100_000
    capabilities: ClassVar[frozenset[str]] = frozenset({"refit", "central_difference"})

    def __post_init__(self) -> None:
        _nonnegative(self.l1_penalty, "l1_penalty")
        _nonnegative(self.l2_penalty, "l2_penalty")
        _options(self.fit_intercept, self.tolerance, self.max_iter)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        n0: int | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> FittedSparseModel:
        """Fit X(case, feature), y(case) with absolute weights and frozen n0.

        Returns immutable parameters, exact support and convergence diagnostics.
        No feature standardization or automatic tuning is performed.
        """
        return _fit_sparse(
            X,
            y,
            weights=weights,
            n0=n0,
            feature_names=feature_names,
            fit_intercept=self.fit_intercept,
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            tolerance=self.tolerance,
            max_iter=self.max_iter,
        )
