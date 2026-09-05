"""Fixed-threshold Huber regression with explicit numerical reference refits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray, NumericalError, ObjectiveSpec
from .models import RidgeRegressor, _immutable, _matrix
from .sparse import _dependency, _nonnegative, _options, _prepare


@dataclass(frozen=True, eq=False)
class FittedHuberModel:
    """Immutable fixed-scale Huber fit with numerical convergence diagnostics.

    Parameters are optional intercept first, followed by slopes. The fixed Huber
    delta has response units; no residual scale is estimated or silently refit.
    Diagnostics identify the optimizer, score residual, curvature rank and the
    residuals in the quadratic versus linear regions.
    """

    parameters: FloatArray
    parameter_names: tuple[str, ...]
    objective: ObjectiveSpec
    objective_value: float
    _residuals: FloatArray = field(repr=False)
    _diagnostic_items: tuple[tuple[str, Any], ...] = field(repr=False)

    @property
    def coefficients(self) -> FloatArray:
        """Return immutable slopes, excluding the optional intercept."""
        return self.parameters[int(self.objective.fit_intercept) :]

    @property
    def intercept(self) -> float:
        """Return the fitted unpenalized intercept or zero."""
        return float(self.parameters[0]) if self.objective.fit_intercept else 0.0

    @property
    def residuals(self) -> FloatArray:
        """Return immutable response-minus-prediction training residuals."""
        return self._residuals

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return a fresh copy of scalar convergence and objective diagnostics."""
        return dict(self._diagnostic_items)

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict from a finite feature matrix of shape (case, feature)."""
        design = _matrix(X)
        if design.shape[1] != self.coefficients.size:
            raise ValueError("Prediction feature count differs from the Huber fit.")
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = design @ self.coefficients + self.intercept
        if not np.isfinite(prediction).all():
            raise NumericalError("Huber prediction is nonfinite; rescale inputs explicitly.")
        return prediction


@dataclass(frozen=True)
class HuberRegressor:
    """Huber regression with a fixed residual threshold and optional ridge.

    Parameters
    ----------
    delta : float, default 1.35
        Positive fixed threshold in response units. Loss is half squared error
        for abs(residual) <= delta and delta*(abs(residual)-delta/2) otherwise.
    penalty : float, default 0
        Canonical lambda2 multiplying half the squared slope norm.
    fit_intercept : bool, default True
        Include an unpenalized intercept.
    tolerance : float, default 1e-8
        Optimizer gradient tolerance. The explicit post-fit stationarity check
        uses 10*tolerance times max(1, initial gradient infinity norm).
    max_iter : int, default 10000
        Maximum deterministic L-BFGS-B iterations.

    Notes
    -----
    Minimizes sum(weights * Huber(residual))/n0 plus the ridge penalty. Scale is
    fixed, not jointly estimated. Numerical refits and central differences are
    supported; implicit derivatives are not advertised. A bounded residual score
    does not make leverage-point influence bounded in the predictors.
    """

    delta: float = 1.35
    penalty: float = 0.0
    fit_intercept: bool = True
    tolerance: float = 1e-8
    max_iter: int = 10_000
    capabilities: ClassVar[frozenset[str]] = frozenset({"refit", "central_difference"})

    def __post_init__(self) -> None:
        _nonnegative(self.delta, "delta", positive=True)
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
    ) -> FittedHuberModel:
        """Fit X(case, feature), y(case), absolute weights and fixed baseline n0.

        Returns an immutable fitted snapshot. Optimization failures, nonfinite
        values, failed stationarity checks, or uncertified unique optima raise
        NumericalError. Feature names label columns without an intercept.
        """
        X_array, response, w, count, names = _prepare(
            X, y, weights, n0, feature_names, self.fit_intercept
        )
        design = (
            np.column_stack([np.ones(len(response)), X_array]) if self.fit_intercept else X_array
        )
        p = design.shape[1]
        start = int(self.fit_intercept)
        penalty_diagonal = np.ones(p)
        if self.fit_intercept:
            penalty_diagonal[0] = 0

        def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
            with np.errstate(over="ignore", invalid="ignore"):
                residual = response - design @ parameters
                absolute = np.abs(residual)
                quadratic = np.minimum(absolute, self.delta)
                losses = quadratic**2 / 2 + self.delta * (absolute - quadratic)
                value = float(
                    w @ losses / count
                    + self.penalty * (parameters[start:] @ parameters[start:]) / 2
                )
                gradient = -(design.T @ (w * np.clip(residual, -self.delta, self.delta))) / count
                gradient += self.penalty * penalty_diagonal * parameters
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                raise NumericalError(
                    "Huber objective or score overflowed; rescale inputs explicitly."
                )
            return value, gradient

        initial = (
            RidgeRegressor(self.penalty, self.fit_intercept)
            .fit(X_array, response, weights=w, n0=count)
            .parameters
        )
        initial_gradient_scale = max(1.0, float(np.abs(objective(initial)[1]).max()))
        optimize = _dependency("scipy.optimize")
        scipy = _dependency("scipy")
        solved = optimize.minimize(
            objective,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "gtol": self.tolerance,
                "ftol": np.finfo(float).eps,
                "maxiter": self.max_iter,
                "maxls": 50,
            },
        )
        parameters = np.asarray(solved.x, dtype=float)
        value, gradient = objective(parameters)
        score_norm = float(np.abs(gradient).max())
        stationarity_tolerance = 10 * self.tolerance * initial_gradient_scale
        if not solved.success or score_norm > stationarity_tolerance:
            raise NumericalError(
                f"Huber optimizer failed convergence: {solved.message}; score {score_norm:g}, required {stationarity_tolerance:g}. Increase max_iter or rescale explicitly."
            )
        residual = response - design @ parameters
        quadratic_region = np.abs(residual) < self.delta
        curvature_design = design * np.sqrt(w * quadratic_region)[:, None]
        if self.penalty:
            curvature_design = np.vstack(
                [curvature_design, np.diag(np.sqrt(count * self.penalty) * penalty_diagonal)]
            )
        rank = int(np.linalg.matrix_rank(curvature_design))
        if rank < p:
            raise NumericalError(
                "Huber optimum uniqueness could not be certified: quadratic-region curvature is rank deficient. Use explicit positive ridge or revise the data."
            )
        singular = np.linalg.svd(curvature_design, compute_uv=False)
        diagnostics: dict[str, Any] = {
            "solver": "scipy.optimize.minimize_L-BFGS-B",
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "optimizer_success": bool(solved.success),
            "optimizer_status": int(solved.status),
            "optimizer_message": str(solved.message),
            "n_iter": int(solved.nit),
            "n_function_evaluations": int(solved.nfev),
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "stationarity_residual_norm": score_norm,
            "stationarity_tolerance": stationarity_tolerance,
            "curvature_rank": rank,
            "curvature_design_condition": float(singular[0] / singular[-1]),
            "quadratic_cases": int(np.count_nonzero(quadratic_region & (w > 0))),
            "linear_cases": int(np.count_nonzero(~quadratic_region & (w > 0))),
            "threshold_margin": float(np.min(np.abs(np.abs(residual[w > 0]) - self.delta))),
            "huber_delta": self.delta,
            "scale_policy": "fixed_threshold_in_response_units",
            "n0": count,
            "weight_sum": float(w.sum()),
            "penalty": self.penalty,
            "objective_value": value,
            "standardization": "none",
            "implicit_supported": False,
        }
        return FittedHuberModel(
            _immutable(parameters),
            (("intercept",) if self.fit_intercept else ()) + names,
            ObjectiveSpec(
                count,
                self.penalty,
                self.fit_intercept,
                loss="huber_fixed_delta",
                huber_delta=self.delta,
            ),
            value,
            _immutable(residual),
            tuple(diagnostics.items()),
        )
