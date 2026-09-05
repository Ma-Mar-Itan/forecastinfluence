"""Rolling source budgets account for repeated fitting without changing effects."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecastinfluence import (
    AddToValues,
    BudgetError,
    ForecastValue,
    InfluenceRequest,
    LagFeatures,
    RawObservationWindow,
    RecursiveForecaster,
    RidgeRegressor,
    RollingInfluenceStudy,
)


def rolling():
    return RollingInfluenceStudy(
        forecaster=RecursiveForecaster(RidgeRegressor(0.1), LagFeatures([1])),
        origins=[8, 11],
        horizons=[1, 2],
        window=RawObservationWindow(length=6),
    ).fit(y=pd.Series(np.sin(np.arange(12)), name="signal"))


def test_plan_eligibility_and_batched_refit_budget():
    study = rolling()
    selection = study.sources(unit="observation").last(3)
    request = InfluenceRequest(selection, AddToValues(0.3), ForecastValue(), "effect", "refit")
    plan = study.plan(request)
    assert plan.baseline_fits == 2
    assert plan.expected_refits == 6
    assert plan.eligible_sources == 3
    assert [r["status"] for r in plan.eligibility] == ["not_observed"] * 3 + ["ok"] * 3
    # Two batches repeat the two origins: 4 baseline plus at most 6 refits.
    with pytest.raises(BudgetError):
        list(study.iter_batches(request, batch_size=2, max_fits=9))
    batches = list(study.iter_batches(request, batch_size=2, max_fits=10))
    combined = xr.concat(
        [b.dataset for b in batches],
        dim="source",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
    )
    xr.testing.assert_equal(combined, study.run(request).dataset)
    with pytest.raises(BudgetError):
        list(study.iter_batches(request, batch_size=2, max_bytes=1))


def test_group_batches_remain_simultaneous_and_invalid_sizes_fail():
    study = rolling()
    request = InfluenceRequest(
        study.sources(unit="observation").last(2).as_group("event"),
        AddToValues(0.1),
        ForecastValue(),
        "effect",
        "refit",
    )
    results = list(study.iter_batches(request, batch_size=1))
    assert len(results) == 1
    xr.testing.assert_equal(results[0].dataset, study.run(request).dataset)
    with pytest.raises(ValueError):
        list(study.iter_batches(request, batch_size=0))


def test_old_source_plan_establishes_exclusion():
    study = rolling()
    request = InfluenceRequest(
        study.sources(unit="observation").at(0), AddToValues(1), ForecastValue(), "effect", "refit"
    )
    plan = study.plan(request)
    assert plan.eligible_sources == 0
    assert all(r["status"] == "structural_zero" for r in plan.eligibility)
    assert np.equal(study.run(request).effect, 0).all()
