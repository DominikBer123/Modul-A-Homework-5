function y = pendulum_process(u, Ts)
% PENDULUM_PROCESS Simulate the nonlinear pendulum process.
%
%   y = pendulum_process(u, Ts)
%   pendulum_process([], 0)
%
% The function keeps an internal process state. Call it with an empty
% input to reset the state. The returned matrix has noisy measured
% columns [angle_rad, angular_velocity_rad_s].

    persistent state

    p = pendulum_parameters();
    if isempty(state) || isempty(u)
        state = initial_state(p);
        if isempty(u)
            y = [];
            return;
        end
    end

    u = u(:);
    N = numel(u);
    dts = sample_times(Ts, N);
    y = zeros(N, 2);

    for k = 1:N
        uk = min(max(u(k), p.u_min), p.u_max);
        state = advance_state(state, uk, dts(k), p);
        y(k, :) = measured_output(state, p);
    end
end

function p = pendulum_parameters()
    p.J = 0.05;
    p.L = 0.2;
    p.g = 9.81;
    p.c = 0.1;
    p.c_visc = 1e-15;
    p.mu_s = 0.05;
    p.mu_k = 0.04;
    p.omega_threshold = 1e-6;
    p.lim_min = 10 * pi / 180;
    p.lim_max = 100 * pi / 180;
    p.restitution = 0.5;
    p.wall_epsilon = 1e-5;
    p.max_substep = 1e-3;
    p.u_min = 0.0;
    p.u_max = 3.0;
    p.theta_noise = 2e-3;
    p.omega_noise = 1e-4;
    p.theta_resolution = 1e-2;
end

function state = initial_state(p)
    state.theta = (p.lim_min + p.lim_max) / 2;
    state.omega = 0.0;
    state.stick_mode = true;
end

function dts = sample_times(Ts, N)
    if isscalar(Ts)
        dts = repmat(Ts, N, 1);
    else
        tVec = Ts(:);
        if numel(tVec) < 2
            error('Time vector must have at least 2 points.');
        end
        dts = diff(tVec);
        if numel(dts) == N - 1
            dts(end + 1) = dts(end);
        end
        if numel(dts) ~= N
            error('Time vector length is inconsistent with input length.');
        end
    end
end

function state = advance_state(state, u, dt, p)
    nSub = max(1, ceil(dt / p.max_substep));
    h = dt / nSub;
    for i = 1:nSub
        state = advance_substep(state, u, h, p);
    end
end

function state = advance_substep(state, u, h, p)
    netTorque = net_applied_torque(state.theta, state.omega, u, p);
    if state.stick_mode
        if abs(netTorque) <= p.mu_s * p.J
            state.omega = 0.0;
            return;
        end
        state.stick_mode = false;
        state.omega = sign_nonzero(netTorque) * p.omega_threshold;
    end

    omegaBefore = state.omega;
    x = rk4_step([state.theta; state.omega], u, h, p);
    state.theta = x(1);
    state.omega = x(2);

    state = apply_limits(state, p);

    crossedZero = omegaBefore ~= 0 && sign(omegaBefore) ~= sign(state.omega);
    if crossedZero || abs(state.omega) <= p.omega_threshold
        netTorque = net_applied_torque(state.theta, state.omega, u, p);
        if abs(netTorque) <= p.mu_s * p.J
            state.omega = 0.0;
            state.stick_mode = true;
        end
    end
end

function xNext = rk4_step(x, u, h, p)
    k1 = dynamics_slip(x, u, p);
    k2 = dynamics_slip(x + 0.5 * h * k1, u, p);
    k3 = dynamics_slip(x + 0.5 * h * k2, u, p);
    k4 = dynamics_slip(x + h * k3, u, p);
    xNext = x + (h / 6) * (k1 + 2*k2 + 2*k3 + k4);
end

function dx = dynamics_slip(x, u, p)
    theta = x(1);
    omega = x(2);
    invJ = 1.0 / p.J;
    dtheta = omega;
    domega = -(p.g / p.L) * sin(theta) ...
           + invJ * u ...
           - (p.c * invJ) * omega ...
           - (p.c_visc * invJ) * omega ...
           - (p.mu_k * invJ) * sign_nonzero(omega);
    dx = [dtheta; domega];
end

function state = apply_limits(state, p)
    if state.theta < p.lim_min
        state.theta = p.lim_min + p.wall_epsilon;
        state.omega = -state.omega * p.restitution;
        if state.omega < 0
            state.omega = 0.0;
        end
    elseif state.theta > p.lim_max
        state.theta = p.lim_max - p.wall_epsilon;
        state.omega = -state.omega * p.restitution;
        if state.omega > 0
            state.omega = 0.0;
        end
    end
end

function tau = net_applied_torque(theta, omega, u, p)
    gravityTorque = -(p.g / p.L) * sin(theta) * p.J;
    viscousTorque = -p.c * omega;
    tau = u + gravityTorque + viscousTorque;
end

function y = measured_output(state, p)
    thetaMeasured = state.theta + randn() * p.theta_noise;
    thetaMeasured = round(thetaMeasured / p.theta_resolution) * p.theta_resolution;
    omegaMeasured = state.omega + randn() * p.omega_noise;
    y = [thetaMeasured, omegaMeasured];
end

function s = sign_nonzero(x)
    if x > 0
        s = 1;
    elseif x < 0
        s = -1;
    else
        s = 0;
    end
end
