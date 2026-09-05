"""Native weighted linear regression under the fixed-baseline objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .core import FloatArray, NumericalError, ObjectiveSpec


def _immutable(values: ArrayLike) -> FloatArray:
    """Copy into immutable storage, including protection against setflags."""
    array = np.asarray(values, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _matrix(X: ArrayLike) -> FloatArray:
    if np.iscomplexobj(X):
        raise ValueError("X must contain real values; complex predictors are unsupported.")
    array = np.array(X, dtype=np.float64, copy=True)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("X must be a finite two-dimensional matrix excluding the intercept.")
    return array


@dataclass(frozen=True, eq=False)
class FittedLinearModel:
    """Immutable linear fit and its fixed-denominator derivative operator.

    Attributes
    ----------
    parameters : ndarray of shape (n_parameters,)
        Intercept first when fitted, followed by feature coefficients.
    parameter_names : tuple of str
        Names in the same order as ``parameters``.
    objective : ObjectiveSpec
        Canonical loss, regularization and frozen baseline denominator.
    objective_value : float
        Weighted half squared error divided by n0 plus the ridge penalty.
    diagnostics : dict
        Fresh copy of rank, conditioning, objective and residual diagnostics.
        The condition number refers to the augmented least-squares design;
        its square is the Hessian condition estimate.

    Notes
    -----
    Created by :meth:`OLSRegressor.fit` or :meth:`RidgeRegressor.fit`.
    Parameters and retained fitting arrays own immutable copies of their data.
    No standardization, pseudoinverse fallback, or hidden damping is applied.
    """

    parameters: FloatArray
    parameter_names: tuple[str, ...]
    objective: ObjectiveSpec
    objective_value: float
    _X: FloatArray = field(repr=False)
    _residuals: FloatArray = field(repr=False)
    _R: FloatArray = field(repr=False)
    _diagnostic_items: tuple[tuple[str, Any], ...] = field(repr=False)
    _weights: FloatArray = field(repr=False)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return an independent copy of the fit's numerical diagnostics."""
        return dict(self._diagnostic_items)

    @property
    def coefficients(self) -> FloatArray:
        """Return immutable slope coefficients in feature order."""
        return self.parameters[int(self.objective.fit_intercept) :]

    @property
    def intercept(self) -> float:
        """Return the fitted intercept, or zero when no intercept is fitted."""
        return float(self.parameters[0]) if self.objective.fit_intercept else 0.0

    @property
    def residuals(self) -> FloatArray:
        """Return immutable response-minus-prediction training residuals."""
        return self._residuals

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict responses from a finite feature matrix.

        Parameters
        ----------
        X : array-like of shape (n_cases, n_features)
            Features excluding the intercept, in the fitted column order.

        Returns
        -------
        ndarray of shape (n_cases,)
            Predicted responses.
        """
        design = _matrix(X)
        if design.shape[1] != self.coefficients.size:
            raise ValueError("Prediction feature count differs from the fitted design.")
        with np.errstate(over="ignore", invalid="ignore"):
            result = design @ self.coefficients + self.intercept
        if not np.all(np.isfinite(result)):
            raise NumericalError("Prediction overflowed; rescale features explicitly.")
        return result

    def weight_derivative(self, indices: Sequence[int]) -> FloatArray:
        """Differentiate parameters with respect to absolute case weights.

        Parameters
        ----------
        indices : sequence of int
            Zero-based fitted case positions, in the desired output order.
            Repeated indices are allowed; an empty sequence gives no columns.

        Returns
        -------
        ndarray of shape (n_parameters, len(indices))
            Parameter derivatives evaluated at this fit's weights. At baseline
            weights one, this is ``H^{-1} x_i residual_i / n0``. A full case
            deletion is a finite intervention, not this local derivative.

        Notes
        -----
        Uses two solves with the augmented-design QR factor. No inverse or
        pseudoinverse is constructed. At zero weights this is the smooth
        extension's derivative; only nonnegative perturbations are admissible.
        """
        selected = np.asarray(indices)
        if selected.ndim != 1 or (selected.size and selected.dtype.kind not in "iu"):
            raise ValueError("indices must be a one-dimensional sequence of integers.")
        if selected.size and (np.any(selected < 0) or np.any(selected >= self._X.shape[0])):
            raise IndexError("A case index is outside the fitted case range.")
        selected = selected.astype(np.intp)
        rhs = self._X[selected].T * self._residuals[selected]
        try:
            intermediate = np.linalg.solve(self._R.T, rhs)
            derivative = np.linalg.solve(self._R, intermediate)
        except np.linalg.LinAlgError as exc:
            raise NumericalError(
                "Derivative factorization failed; no fallback was applied."
            ) from exc
        if not np.all(np.isfinite(derivative)):
            raise NumericalError(
                "Weight derivative is nonfinite; inspect conditioning diagnostics."
            )
        return derivative


def _fit(
    X: ArrayLike,
    y: ArrayLike,
    *,
    weights: ArrayLike | None,
    n0: int | None,
    feature_names: Sequence[str] | None,
    fit_intercept: bool,
    penalty: float,
) -> FittedLinearModel:
    features = _matrix(X)
    n, n_features = features.shape
    if np.iscomplexobj(y) or (weights is not None and np.iscomplexobj(weights)):
        raise ValueError("y and weights must be real-valued; complex inputs are unsupported.")
    response = np.array(y, dtype=np.float64, copy=True)
    if not n or response.shape != (n,) or not np.all(np.isfinite(response)):
        raise ValueError("y must be a finite vector matching nonempty X rows.")
    denominator = n if n0 is None else n0
    if (
        isinstance(denominator, (bool, np.bool_))
        or not isinstance(denominator, (int, np.integer))
        or denominator <= 0
    ):
        raise ValueError("n0 must be a positive integer baseline case count.")
    sample_weights = (
        np.ones(n) if weights is None else np.array(weights, dtype=np.float64, copy=True)
    )
    if (
        sample_weights.shape != (n,)
        or not np.all(np.isfinite(sample_weights))
        or np.any(sample_weights < 0)
        or not np.any(sample_weights > 0)
    ):
        raise ValueError(
            "weights must be finite, nonnegative, match y and include a positive weight."
        )
    names = (
        tuple(f"x{i}" for i in range(n_features)) if feature_names is None else tuple(feature_names)
    )
    if (
        len(names) != n_features
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or (fit_intercept and "intercept" in names)
    ):
        raise ValueError(
            "feature_names must contain unique nonempty names, excluding reserved 'intercept'."
        )
    design = np.column_stack((np.ones(n), features)) if fit_intercept else features
    p = design.shape[1]
    if p == 0:
        raise ValueError("A fit requires at least one feature or an intercept.")
    slope_start = int(fit_intercept)
    with np.errstate(over="ignore", invalid="ignore"):
        root_weights = np.sqrt(sample_weights)
        augmented = design * root_weights[:, None]
        target = response * root_weights
        if penalty:
            penalty_rows = np.zeros((n_features, p))
            penalty_rows[:, slope_start:] = np.eye(n_features) * (
                np.sqrt(float(denominator)) * np.sqrt(penalty)
            )
            augmented = np.vstack((augmented, penalty_rows))
            target = np.concatenate((target, np.zeros(n_features)))
    if not np.all(np.isfinite(augmented)) or not np.all(np.isfinite(target)):
        raise NumericalError("Weighted augmented design overflowed; rescale inputs explicitly.")
    try:
        parameters, _, rank, singular_values = np.linalg.lstsq(augmented, target, rcond=None)
        if rank < p:
            raise NumericalError(
                f"Fit is not numerically unique: augmented design rank {rank} < {p}. No pseudoinverse fallback."
            )
        _, factor = np.linalg.qr(augmented, mode="reduced")
    except np.linalg.LinAlgError as exc:
        raise NumericalError(
            "Augmented least-squares factorization failed; no fallback was applied."
        ) from exc
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = response - design @ parameters
        loss = float(np.sum(sample_weights * residuals**2) / (2 * denominator))
        regularization = float(penalty * np.sum(parameters[slope_start:] ** 2) / 2)
        augmented_residual = augmented @ parameters - target
        gradient = augmented.T @ augmented_residual / denominator
        gradient_scale = float(
            np.linalg.norm(augmented, ord="fro") * np.linalg.norm(augmented_residual) / denominator
        )
        stationarity = float(np.linalg.norm(gradient))
    if (
        not np.all(np.isfinite(parameters))
        or not np.all(np.isfinite(residuals))
        or not np.isfinite(loss + regularization)
        or not np.isfinite(stationarity)
    ):
        raise NumericalError("Fit diagnostics are nonfinite; rescale inputs explicitly.")
    condition = float(singular_values[0] / singular_values[-1])
    diagnostics: dict[str, Any] = {
        "solver": "numpy.linalg.lstsq_augmented_svd",
        "derivative_solver": "augmented_qr_two_solves",
        "numpy_version": np.__version__,
        "rank": int(rank),
        "n_parameters": p,
        "n_cases": n,
        "n_positive_weights": int(np.count_nonzero(sample_weights)),
        "condition_number": condition,
        "hessian_condition_estimate": condition**2,
        "ill_conditioned": condition > 1 / np.sqrt(np.finfo(float).eps),
        "rank_tolerance": float(np.finfo(float).eps * max(augmented.shape) * singular_values[0]),
        "smallest_singular_value": float(singular_values[-1]),
        "weighted_residual_norm": float(np.linalg.norm(root_weights * residuals)),
        "stationarity_residual_norm": stationarity,
        "relative_stationarity_residual": stationarity / max(gradient_scale, np.finfo(float).tiny),
        "objective_loss": loss,
        "objective_penalty": regularization,
        "objective_value": loss + regularization,
        "n0": int(denominator),
        "penalty": penalty,
        "standardization": "none",
        "pseudoinverse": False,
        "damping": 0.0,
    }
    return FittedLinearModel(
        _immutable(parameters),
        (("intercept",) if fit_intercept else ()) + names,
        ObjectiveSpec(int(denominator), penalty, fit_intercept),
        loss + regularization,
        _immutable(design),
        _immutable(residuals),
        _immutable(factor),
        tuple(diagnostics.items()),
        _immutable(sample_weights),
    )


@dataclass(frozen=True)
class OLSRegressor:
    """Ordinary least squares with optional unpenalized intercept.

    Parameters
    ----------
    fit_intercept : bool, default True
        Include an intercept before feature coefficients.

    Notes
    -----
    Nonunique fits raise ``NumericalError``. No automatic feature scaling is used.
    """

    fit_intercept: bool = True
    capabilities: ClassVar[frozenset[str]] = frozenset({"refit", "implicit", "central_difference"})

    def __post_init__(self) -> None:
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be a bool.")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        n0: int | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> FittedLinearModel:
        """Fit the canonical weighted half-squared-error objective.

        Parameters
        ----------
        X : array-like of shape (n_cases, n_features)
            Finite predictors, excluding an intercept column.
        y : array-like of shape (n_cases,)
            Finite training responses.
        weights : array-like of shape (n_cases,), optional
            Absolute nonnegative case weights; defaults to all ones.
        n0 : int, optional
            Positive baseline denominator; defaults to n_cases. Keep the
            original n0 when intervening on case weights or retaining rows.
        feature_names : sequence of str, optional
            Unique feature names; defaults to x0, x1, and so forth.

        Returns
        -------
        FittedLinearModel
            Immutable parameters, diagnostics and derivative operator.
        """
        return _fit(
            X,
            y,
            weights=weights,
            n0=n0,
            feature_names=feature_names,
            fit_intercept=self.fit_intercept,
            penalty=0.0,
        )


@dataclass(frozen=True)
class RidgeRegressor:
    """Ridge regression with an unpenalized optional intercept.

    Parameters
    ----------
    penalty : float, default 1.0
        Canonical lambda2 in ``sum(w * residual**2)/(2*n0) +
        lambda2 * sum(slopes**2)/2``. A summed-loss solver's alpha equals
        ``n0 * penalty``, not ``penalty``.
    fit_intercept : bool, default True
        Include an intercept before feature coefficients.
    """

    penalty: float = 1.0
    fit_intercept: bool = True
    capabilities: ClassVar[frozenset[str]] = frozenset({"refit", "implicit", "central_difference"})

    def __post_init__(self) -> None:
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be a bool.")
        if isinstance(self.penalty, (bool, np.bool_)) or not isinstance(
            self.penalty, (int, float, np.integer, np.floating)
        ):
            raise ValueError("penalty must be a finite nonnegative scalar.")
        value = float(self.penalty)
        if not np.isfinite(value) or value < 0:
            raise ValueError("penalty must be a finite nonnegative scalar.")
        object.__setattr__(self, "penalty", value)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        n0: int | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> FittedLinearModel:
        """Fit ridge via weighted augmented least squares.

        Parameters
        ----------
        X : array-like of shape (n_cases, n_features)
            Finite predictors, excluding an intercept column.
        y : array-like of shape (n_cases,)
            Finite training responses.
        weights : array-like of shape (n_cases,), optional
            Absolute nonnegative case weights; defaults to all ones.
        n0 : int, optional
            Frozen positive baseline denominator; defaults to n_cases.
        feature_names : sequence of str, optional
            Unique names for the feature columns.

        Returns
        -------
        FittedLinearModel
            Immutable fit; nonunique designs raise ``NumericalError``.
        """
        return _fit(
            X,
            y,
            weights=weights,
            n0=n0,
            feature_names=feature_names,
            fit_intercept=self.fit_intercept,
            penalty=self.penalty,
        )
