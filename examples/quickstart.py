"""Quickstart (synthetic data, about two seconds):
1) operational modal identification with correlation-domain Hankel-DMD;
2) one Hankel-DMD model that reconstructs the hidden channels of an unseen record from a single sensor, with the
   projection estimator and its Kalman form.

Usage: python examples/quickstart.py
"""
import numpy as np

from eftwin import KalmanEstimator, ProjectionEstimator, fit_hankel_dmd, identify_modes
from eftwin.synthetic import tower_response
from eftwin.validation import r2_channels

dt = 0.05                                                         # 20 Hz
modes = ((0.5, 0.02), (1.6, 0.02), (3.7, 0.015))                 # true (frequency Hz, damping ratio)
train = [tower_response(6000, dt, modes, seed=s)[0] for s in (1, 2, 3)]    # three records, 6 channels each
test, _ = tower_response(6000, dt, modes, seed=4)

# 1) identification (each record a separate segment; correlation domain is used for damping estimation)
f, zeta, E, mu, _ = identify_modes(train, d=60, r=10, dt=dt, lags=400)
for lo, hi in ((0.3, 0.8), (1.2, 2.0), (3.0, 4.5)):
    m = np.where((mu.imag > 0) & (f > lo) & (f < hi))[0]
    j = m[np.argmax(E[m])]                                        # pole of largest modal energy in the band
    print(f"mode in {lo}-{hi} Hz: f = {f[j]:.3f} Hz, damping = {100 * zeta[j]:.2f} %")

# 2) virtual sensing: one model for every layout; here only the top sensor (channel 5) is measured
model = fit_hankel_dmd(train, d=60, r=10)
active, hidden = [5], [0, 1, 2, 3, 4]
start = 60                                                        # skip the start-up of the estimators
for name, est in (("projection", ProjectionEstimator(model, active, alpha=0.01)), ("Kalman form", KalmanEstimator(model, active))):
    y = est.estimate(test)
    print(f"{name:12s} R^2 of the hidden channels: {np.round(r2_channels(test[start:, hidden].T, y[start:, hidden].T), 3)}")
