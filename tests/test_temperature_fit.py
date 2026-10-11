# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3", "scipy==1.18.1", "pytest==9.1.1"]
# ///
# Run: uv run --frozen pytest tests/test_temperature_fit.py
import numpy as np
from scipy.special import expit
from temperature_fits import Observations, fit_model, predict
from thermal_core import R


def test_multitemperature_fit_recovers_an_unseen_temperature() -> None:
    # Given noise-free training curves generated with known Q, affinity and enthalpy.
    p = np.tile(np.geomspace(0.005, 1.0, 25), 3)
    t = np.repeat([273.0, 298.0, 313.0], 25)
    reference = 298.0
    truth = 4.0 * expit(np.log(p) + 1.0 + 25.0 / R * (1 / t - 1 / reference))
    scale = float(truth.max())
    training = Observations(p, t, truth, np.full(len(p), 1 / np.sqrt(len(p))), scale, reference)
    # When fitting only the training temperatures and predicting333K.
    fit = fit_model(training, thermal=True)
    test_p = np.geomspace(0.005, 1.0, 15)
    test = Observations(test_p, np.full(15, 333.0), np.zeros(15), np.ones(15), scale, reference)
    output = predict(fit.parameters, test)
    expected = 4.0 * expit(np.log(test_p) + 1.0 + 25.0 / R * (1 / 333.0 - 1 / reference))
    # Then the held-out response is recovered without using its labels.
    assert np.max(np.abs(output - expected)) < 1e-7
