% EXAMPLE_PENDULUM_PROCESS  Basic use of pendulum_process.m.

clear; close all; clc;

Ts = 0.01;
Tsim = 20;
N = round(Tsim / Ts);
t = (0:N-1)' * Ts;

u = 1.5 + 0.7 * (sin(0.7 * t) > 0) + 0.25 * sin(2.1 * t);
u = min(max(u, 0), 3);

% Vector simulation.
rng(10);
pendulum_process([], 0);
yVector = pendulum_process(u, Ts);

% Step-by-step simulation. With the same random seed, this should match
% the vector simulation up to numerical roundoff.
rng(10);
pendulum_process([], 0);
yStep = zeros(N, 2);
for k = 1:N
    yStep(k, :) = pendulum_process(u(k), Ts);
end

fprintf('Max vector/step angle difference: %.3e rad\n', ...
        max(abs(yVector(:, 1) - yStep(:, 1))));
fprintf('Max vector/step velocity difference: %.3e rad/s\n', ...
        max(abs(yVector(:, 2) - yStep(:, 2))));

figure('Name', 'Pendulum process example', 'Color', 'w');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(t, u, 'LineWidth', 1.2);
grid on;
ylabel('u [Nm]');
title('Input torque');

nexttile;
plot(t, rad2deg(yVector(:, 1)), 'LineWidth', 1.2);
grid on;
ylabel('\phi [deg]');
title('Measured angle');

nexttile;
plot(t, yVector(:, 2), 'LineWidth', 1.2);
grid on;
xlabel('time [s]');
ylabel('\omega [rad/s]');
title('Measured angular velocity');
