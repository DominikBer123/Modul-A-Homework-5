"""Basic use of pendulum_process.py."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pendulum_process import PendulumProcess, pendulum_process


def main() -> None:
    Ts = 0.01
    Tsim = 20.0
    n_samples = round(Tsim / Ts)
    t = np.arange(n_samples) * Ts

    u = 1.5 + 0.7 * (np.sin(0.7 * t) > 0) + 0.25 * np.sin(2.1 * t)
    u = np.clip(u, 0.0, 3.0)

    # Vector simulation.
    y_vector = pendulum_process(u, Ts, reset=True, seed=10)

    # Step-by-step simulation. With the same random seed, this should
    # match the vector simulation up to numerical roundoff.
    process = PendulumProcess(seed=10)
    y_step = np.zeros((n_samples, 2))
    for k, value in enumerate(u):
        y_step[k, :] = process.step(float(value), Ts)

    print(
        "Max vector/step angle difference: "
        f"{np.max(np.abs(y_vector[:, 0] - y_step[:, 0])):.3e} rad"
    )
    print(
        "Max vector/step velocity difference: "
        f"{np.max(np.abs(y_vector[:, 1] - y_step[:, 1])):.3e} rad/s"
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, u, lw=1.2)
    axes[0].set_ylabel("u [Nm]")
    axes[0].set_title("Input torque")
    axes[0].grid(True)

    axes[1].plot(t, np.rad2deg(y_vector[:, 0]), lw=1.2)
    axes[1].set_ylabel("phi [deg]")
    axes[1].set_title("Measured angle")
    axes[1].grid(True)

    axes[2].plot(t, y_vector[:, 1], lw=1.2)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("omega [rad/s]")
    axes[2].set_title("Measured angular velocity")
    axes[2].grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
