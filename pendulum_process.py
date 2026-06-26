"""Plain Python pendulum process matching ``pendulum_process.m``.

The process keeps an internal state. Use ``reset()`` on the class or
call ``pendulum_process([], 0, reset=True)`` to return to the initial
condition. The output has noisy measured columns
``[angle_rad, angular_velocity_rad_s]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


ArrayLike = Union[np.ndarray, list[float], tuple[float, ...], float]


@dataclass
class PendulumParameters:
    J: float = 0.05
    L: float = 0.2
    g: float = 9.81
    c: float = 0.1
    c_visc: float = 1e-15
    mu_s: float = 0.05
    mu_k: float = 0.04
    omega_threshold: float = 1e-6
    lim_min: float = 10.0 * np.pi / 180.0
    lim_max: float = 100.0 * np.pi / 180.0
    restitution: float = 0.5
    wall_epsilon: float = 1e-5
    max_substep: float = 1e-3
    u_min: float = 0.0
    u_max: float = 3.0
    theta_noise: float = 2e-3
    omega_noise: float = 1e-4
    theta_resolution: float = 1e-2


@dataclass
class PendulumState:
    theta: float
    omega: float
    stick_mode: bool


def _sign_nonzero(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


class PendulumProcess:
    """Persistent nonlinear pendulum simulator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.params = PendulumParameters()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Reset the process to its initial state."""
        p = self.params
        self.state = PendulumState(
            theta=(p.lim_min + p.lim_max) / 2.0,
            omega=0.0,
            stick_mode=True,
        )

    def simulate(self, u: ArrayLike, Ts: Union[float, np.ndarray], reset: bool = True) -> np.ndarray:
        """Simulate a scalar or vector input."""
        if reset:
            self.reset()

        u_arr = np.asarray(u, dtype=float).ravel()
        if u_arr.size == 0:
            self.reset()
            return np.empty((0, 2))

        dts = self._sample_times(Ts, u_arr.size)
        y = np.zeros((u_arr.size, 2), dtype=float)

        for k, value in enumerate(u_arr):
            uk = float(np.clip(value, self.params.u_min, self.params.u_max))
            self._advance_state(uk, float(dts[k]))
            y[k, :] = self._measured_output()
        return y

    def step(self, u: float, Ts: float) -> np.ndarray:
        """Apply one input sample and return ``[angle, angular_velocity]``."""
        return self.simulate([u], Ts, reset=False)[0, :]

    def close(self) -> None:
        """Compatibility no-op; no external backend is used."""

    def __enter__(self) -> "PendulumProcess":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _sample_times(Ts: Union[float, np.ndarray], n_samples: int) -> np.ndarray:
        Ts_arr = np.asarray(Ts, dtype=float).ravel()
        if Ts_arr.size == 1:
            return np.full(n_samples, float(Ts_arr[0]))
        if Ts_arr.size < 2:
            raise ValueError("Time vector must have at least 2 points.")
        dts = np.diff(Ts_arr)
        if dts.size == n_samples - 1:
            dts = np.concatenate([dts, dts[-1:]])
        if dts.size != n_samples:
            raise ValueError("Time vector length is inconsistent with input length.")
        return dts

    def _advance_state(self, u: float, dt: float) -> None:
        p = self.params
        n_sub = max(1, int(np.ceil(dt / p.max_substep)))
        h = dt / n_sub
        for _ in range(n_sub):
            self._advance_substep(u, h)

    def _advance_substep(self, u: float, h: float) -> None:
        p = self.params
        s = self.state
        net_torque = self._net_applied_torque(s.theta, s.omega, u)

        if s.stick_mode:
            if abs(net_torque) <= p.mu_s * p.J:
                s.omega = 0.0
                return
            s.stick_mode = False
            s.omega = _sign_nonzero(net_torque) * p.omega_threshold

        omega_before = s.omega
        x_next = self._rk4_step(np.array([s.theta, s.omega]), u, h)
        s.theta = float(x_next[0])
        s.omega = float(x_next[1])

        self._apply_limits()

        crossed_zero = omega_before != 0 and np.sign(omega_before) != np.sign(s.omega)
        if crossed_zero or abs(s.omega) <= p.omega_threshold:
            net_torque = self._net_applied_torque(s.theta, s.omega, u)
            if abs(net_torque) <= p.mu_s * p.J:
                s.omega = 0.0
                s.stick_mode = True

    def _rk4_step(self, x: np.ndarray, u: float, h: float) -> np.ndarray:
        k1 = self._dynamics_slip(x, u)
        k2 = self._dynamics_slip(x + 0.5 * h * k1, u)
        k3 = self._dynamics_slip(x + 0.5 * h * k2, u)
        k4 = self._dynamics_slip(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _dynamics_slip(self, x: np.ndarray, u: float) -> np.ndarray:
        p = self.params
        theta = float(x[0])
        omega = float(x[1])
        inv_j = 1.0 / p.J
        dtheta = omega
        domega = (
            -(p.g / p.L) * np.sin(theta)
            + inv_j * u
            - (p.c * inv_j) * omega
            - (p.c_visc * inv_j) * omega
            - (p.mu_k * inv_j) * _sign_nonzero(omega)
        )
        return np.array([dtheta, domega], dtype=float)

    def _apply_limits(self) -> None:
        p = self.params
        s = self.state
        if s.theta < p.lim_min:
            s.theta = p.lim_min + p.wall_epsilon
            s.omega = -s.omega * p.restitution
            if s.omega < 0:
                s.omega = 0.0
        elif s.theta > p.lim_max:
            s.theta = p.lim_max - p.wall_epsilon
            s.omega = -s.omega * p.restitution
            if s.omega > 0:
                s.omega = 0.0

    def _net_applied_torque(self, theta: float, omega: float, u: float) -> float:
        p = self.params
        gravity_torque = -(p.g / p.L) * np.sin(theta) * p.J
        viscous_torque = -p.c * omega
        return u + gravity_torque + viscous_torque

    def _measured_output(self) -> np.ndarray:
        p = self.params
        theta_measured = self.state.theta + self.rng.normal(0.0, p.theta_noise)
        theta_measured = np.round(theta_measured / p.theta_resolution) * p.theta_resolution
        omega_measured = self.state.omega + self.rng.normal(0.0, p.omega_noise)
        return np.array([theta_measured, omega_measured], dtype=float)


_DEFAULT_PROCESS: Optional[PendulumProcess] = None


def pendulum_process(u: ArrayLike,
                     Ts: Union[float, np.ndarray],
                     reset: bool = False,
                     seed: Optional[int] = None) -> np.ndarray:
    """Persistent function-style interface to the pendulum process."""
    global _DEFAULT_PROCESS
    if _DEFAULT_PROCESS is None or seed is not None:
        _DEFAULT_PROCESS = PendulumProcess(seed=seed)

    u_arr = np.asarray(u, dtype=float).ravel()
    if reset or u_arr.size == 0:
        _DEFAULT_PROCESS.reset()
        if u_arr.size == 0:
            return np.empty((0, 2))

    return _DEFAULT_PROCESS.simulate(u_arr, Ts, reset=False)


if __name__ == "__main__":
    process = PendulumProcess()
    print(process.simulate([2.0, 2.1], Ts=0.05))
