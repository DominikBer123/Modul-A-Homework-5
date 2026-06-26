"""Reference generator for Home Assignment 5.

The trajectory starts with random steps of random duration and then
switches to a smooth random signal. All values stay inside the normal
pendulum operating region [20 deg, 80 deg].
"""

from __future__ import annotations

import numpy as np


class _ReferenceRng:
    """Small Park-Miller RNG mirrored in the MATLAB implementation."""

    def __init__(self, seed: int) -> None:
        self.state = int(seed) % 2147483647
        if self.state <= 0:
            self.state += 2147483646

    def random(self) -> float:
        self.state = (16807 * self.state) % 2147483647
        return self.state / 2147483647.0

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.random()

    def integers(self, low: int, high: int) -> int:
        """Return an integer in [low, high], inclusive."""
        return low + int(np.floor((high - low + 1) * self.random()))


def vaja5_reference(Ts: float,
                    T_sim: float,
                    seed: int = 12,
                    step_fraction: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """Generate the homework reference trajectory.

    Parameters
    ----------
    Ts:
        Sampling time in seconds.
    T_sim:
        Total simulation time in seconds.
    seed:
        Random seed for reproducibility.
    step_fraction:
        Fraction of the trajectory occupied by the random-step part.

    Returns
    -------
    ref:
        Reference angle in radians.
    t:
        Time vector in seconds.
    """
    rng = _ReferenceRng(seed)
    n_samples = int(round(T_sim / Ts))
    t = np.arange(n_samples) * Ts
    ref_deg = np.zeros(n_samples)

    step_end = int(round(np.clip(step_fraction, 0.1, 0.9) * n_samples))
    k = 0
    previous_level = 55.0
    while k < step_end:
        hold_samples = rng.integers(
            max(2, round(3.0 / Ts)),
            max(3, round(9.0 / Ts)),
        )
        level = float(rng.uniform(25.0, 75.0))
        if abs(level - previous_level) < 8.0:
            level = 25.0 + np.mod(level + 18.0, 50.0)
        stop = min(step_end, k + hold_samples)
        ref_deg[k:stop] = level
        previous_level = level
        k = stop

    if step_end < n_samples:
        tau = t[step_end:] - t[step_end]
        duration = max(tau[-1], Ts)
        raw = (
            0.65 * np.sin(2.0 * np.pi * tau / duration
                           + rng.uniform(0, 2*np.pi))
            + 0.30 * np.sin(4.0 * np.pi * tau / duration
                            + rng.uniform(0, 2*np.pi))
            + 0.18 * np.sin(7.0 * np.pi * tau / duration
                            + rng.uniform(0, 2*np.pi))
        )
        raw = raw / max(np.max(np.abs(raw)), 1e-12)
        smooth_deg = 50.0 + rng.uniform(-3.0, 3.0) + 24.0 * raw
        smooth_deg = np.clip(smooth_deg, 22.0, 78.0)

        blend_samples = min(int(round(2.0 / Ts)), smooth_deg.size)
        if blend_samples > 1:
            start_level = ref_deg[step_end - 1] if step_end > 0 else smooth_deg[0]
            alpha = np.linspace(0.0, 1.0, blend_samples)
            smooth_deg[:blend_samples] = (
                (1.0 - alpha) * start_level + alpha * smooth_deg[:blend_samples]
            )

        ref_deg[step_end:] = np.clip(smooth_deg, 20.0, 80.0)

    return np.deg2rad(ref_deg), t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    r, tt = vaja5_reference(0.05, 65.0)
    plt.figure()
    plt.plot(tt, np.rad2deg(r))
    plt.axhline(20, color="0.5", ls=":")
    plt.axhline(80, color="0.5", ls=":")
    plt.xlabel("time [s]")
    plt.ylabel("reference [deg]")
    plt.grid(True)
    plt.show()
