function [ref, t] = vaja5_reference(Ts, Tsim, seed, stepFraction)
% VAJA5_REFERENCE  Generate the Home Assignment 5 reference trajectory.
%
%   [ref, t] = vaja5_reference(Ts, Tsim, seed)
%
% The first part contains random steps with random durations. The second
% part is a smooth random signal. The whole reference stays inside the
% normal pendulum operating region [20 deg, 80 deg].

    if nargin < 3 || isempty(seed)
        seed = 12;
    end
    if nargin < 4 || isempty(stepFraction)
        stepFraction = 0.45;
    end

    N = round(Tsim / Ts);
    t = (0:N-1)' * Ts;
    refDeg = zeros(N, 1);

    stepEnd = round(min(max(stepFraction, 0.1), 0.9) * N);
    k = 1;
    previousLevel = 55.0;
    while k <= stepEnd
        holdSamples = ref_rand_int(max(2, round(3.0 / Ts)), ...
                                   max(3, round(9.0 / Ts)));
        level = ref_rand_uniform(25.0, 75.0);
        if abs(level - previousLevel) < 8.0
            level = 25.0 + mod(level + 18.0, 50.0);
        end
        stopIdx = min(stepEnd, k + holdSamples - 1);
        refDeg(k:stopIdx) = level;
        previousLevel = level;
        k = stopIdx + 1;
    end

    if stepEnd < N
        tau = t(stepEnd+1:end) - t(stepEnd+1);
        duration = max(tau(end), Ts);
        raw = 0.65 * sin(2*pi*tau/duration + 2*pi*ref_rand()) ...
            + 0.30 * sin(4*pi*tau/duration + 2*pi*ref_rand()) ...
            + 0.18 * sin(7*pi*tau/duration + 2*pi*ref_rand());
        raw = raw / max(max(abs(raw)), 1e-12);
        smoothDeg = 50.0 + ref_rand_uniform(-3.0, 3.0) + 24.0 * raw;
        smoothDeg = min(max(smoothDeg, 22.0), 78.0);

        blendSamples = min(round(2.0 / Ts), numel(smoothDeg));
        if blendSamples > 1
            if stepEnd > 0
                startLevel = refDeg(stepEnd);
            else
                startLevel = smoothDeg(1);
            end
            alpha = linspace(0.0, 1.0, blendSamples)';
            smoothDeg(1:blendSamples) = ...
                (1.0 - alpha) * startLevel + alpha .* smoothDeg(1:blendSamples);
        end

        refDeg(stepEnd+1:end) = min(max(smoothDeg, 20.0), 80.0);
    end

    ref = deg2rad(refDeg);

    function x = ref_rand()
        seed = mod(16807 * seed, 2147483647);
        x = seed / 2147483647;
    end

    function x = ref_rand_uniform(low, high)
        x = low + (high - low) * ref_rand();
    end

    function x = ref_rand_int(low, high)
        x = low + floor((high - low + 1) * ref_rand());
    end
end
