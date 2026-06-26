clear
close all
clc

% reset System

pendulum_process_obf([], 0); 


Ts = 0.05;              % Čas vzorčenja (s)
T_sim = 10;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor

% Sine wave parameters for bounds [8, 20]
bias = 50; 
amplitude = 30; 
f = 1;                  % Frequency in Hz (adjust as needed)

% Generate the sine wave
refYDeg = bias + amplitude * sin(2 * pi * f * t_vec);



figure
plot(t_vec,refYDeg)


%% Random Staircase Reference

Ts = 0.05;
T_sim = 100;
N = ceil(T_sim/Ts);
t_vec = (0:N-1)'*Ts;

% Limits (degrees)
minAngle = 30;
maxAngle = 60;

% Step duration limits (seconds)
minHold = 1.0;
maxHold = 3.0;

refYDeg = zeros(N,1);

k = 1;
while k <= N

    % Random angle
    level = minAngle + (maxAngle-minAngle)*rand();

    % Random duration
    holdTime = minHold + (maxHold-minHold)*rand();
    holdSamples = round(holdTime/Ts);

    idx = k:min(k+holdSamples-1,N);
    refYDeg(idx) = level;

    k = idx(end)+1;
end

% Convert to radians
refY = deg2rad(refYDeg);

% Plot
figure
stairs(t_vec,refYDeg,'LineWidth',1.5)
grid on
xlabel('Time [s]')
ylabel('Reference [deg]')
title('Random Staircase Reference')





refY = deg2rad(refYDeg);



%% začetek lab vaje 5
fuzziness =2;

theta_local = [
    0.0170,  0.0474,  1.4574, -0.5858, -0.0114;
    0.0210,  0.0351,  1.5835, -0.6966, -0.0091;
    0.0191,  0.0413,  1.5204, -0.6401, -0.0109;
    0.0204,  0.0361,  1.5810, -0.6942, -0.0094;
    0.0195,  0.0391,  1.5469, -0.6643, -0.0100
];

centersXY = [
    1.7906,  0.8020;
    0.1918,  0.1834;
    2.3567,  1.2609;
    0.8929,  0.3604;
    2.7668,  1.7451
];


% In your main/test script, pack your parameters up before calling the loop:
state.centersXY   = centersXY;
state.theta_local = theta_local;
state.fuzziness   = 2.0; % your fuzzy exponent
state.numRules    = 5;   % your number of clusters/rules

%% Main Receding Horizon Control Loop

% 2. MPC Horizon Configurations

S = 1;                        % Control Horizon (1 decision parameter)
segLengths = 8;               % A single segment of length 6 (sums to H)
H = sum(segLengths);
cfg.lamDu = 0.5;

% Actuator saturation limits (Adjust these to match your actual pendulum bounds)
uMin = 0.0;                 
uMax = 3.0;                 

% 3. Allocate memory for closed-loop history
u_cl = zeros(N, 1);         % Closed-loop control history
y_cl = zeros(N, 1);         % Closed-loop system output history

% Initialize the system with the first couple of true measurements to kickstart NARX lag
y_cl(1) = 0.9100;
y_cl(2) = 0.7700;
u_cl(1) = 0;
u_cl(2) = 0;

% Warm-start initialization for the optimizer vector p (size S x 1)
pWarm = zeros(S, 1);

disp('Starting MPC Closed-Loop Simulation...');

% 4. Simulation Loop (Receding Horizon)
for k = 3:N
    % Define the target reference window for the current horizon H
    % Ensure we don't exceed vector bounds near the end of the simulation
    if k + H - 1 <= N
        ref_window = refY(k : k + H - 1);
    else
        ref_window = [refY(k:end); ones(H - (N - k + 1), 1) * refY(end)];
    end
    
    % Get past feedback history states
    yKm1 = y_cl(k-1);
    yKm2 = y_cl(k-2);
    uKm1 = u_cl(k-1);
    
    % --- Call your MPC solver function ---
    pOpt = solve_mpc_step([], ref_window, segLengths, ...
                          yKm1, yKm2, uKm1, state, pWarm, cfg, ...
                          uMin, uMax);
                      
    % Extract the VERY FIRST control choice from the optimized sequence
    % Because we expanded segments, pOpt(1) governs the first segment
    u_current = pOpt(1);
    u_cl(k) = u_current;
    
    % Update your warm start guess for the next iteration (time step k+1)
    pWarm = pOpt; 
    
    %Calculates the next value from the system
    y_new = pendulum_process_obf(u_cl(k), Ts);
    y_cl(k) = y_new(1,1); % Extract the current true system state
    
    % Simple progress tracker
    if mod(k, 200) == 0
        fprintf('Progress: %.1f%%\n', (k/N)*100);
    end
end

disp('Simulation finished! Plotting results...');
%% Plot MPC Results

figure('Name','MPC Closed Loop Results','NumberTitle','off');

%----------------------------------------------------------
% Output Tracking
%----------------------------------------------------------
subplot(3,1,1)
plot(t_vec, rad2deg(refY),'r--','LineWidth',1.5); hold on;
plot(t_vec, rad2deg(y_cl),'b','LineWidth',1.5);
grid on;
xlabel('Time [s]')
ylabel('Angle [deg]')
title('Reference Tracking')
legend('Reference','Pendulum Output','Location','best')

%----------------------------------------------------------
% Control Input
%----------------------------------------------------------
subplot(3,1,2)
stairs(t_vec,u_cl,'k','LineWidth',1.5)
grid on
xlabel('Time [s]')
ylabel('Control Input')
title('MPC Control Signal')

%----------------------------------------------------------
% Tracking Error
%----------------------------------------------------------
subplot(3,1,3)
plot(t_vec,rad2deg(refY - y_cl),'m','LineWidth',1.5)
grid on
xlabel('Time [s]')
ylabel('Error [deg]')
title('Tracking Error')
%% 5. Plot the tracking performance
figure;
subplot(2,1,1);
plot(t_vec, refY, 'r--', 'LineWidth', 1.5); hold on;
plot(t_vec, y_cl, 'b-', 'LineWidth', 1.5);
grid on;
legend('Reference (refY)', 'System Output (y\_cl)');
title('Nonlinear MPC Setpoint Tracking Performance');
ylabel('Output (y)');

subplot(2,1,2);
plot(t_vec, u_cl, 'k-', 'LineWidth', 1.5);
grid on;
legend('Control Action (u\_cl)');
xlabel('Time (s)');
ylabel('Input (u)');


%% Return the uSeq required to do forcasting
function uSeq = expand_segments(p, segLengths) %#ok<DEFNU>
% EXPAND_SEGMENTS   Turn p into the full length-H input sequence.
    uSeq = [];
    for lenIndex = 1:length(segLengths)
        uSegment = ones(segLengths(lenIndex),1) * p(lenIndex);
        uSeq = [uSeq; uSegment];
    end
end

%% makes the prediction for H samples
function yhat = simulate_horizon_mpc(uSeq, yKm1, yKm2, uKm1, centersXY, theta_local, fuzziness, numRules)
    H = length(uSeq);
    yhat = zeros(H,1);
    
    for h_idx = 1:H
        % --- Current Candidate Inputs ---
        u1 = uSeq(h_idx);
        if h_idx == 1
            u2 = uKm1;
        else
            u2 = uSeq(h_idx-1);
        end
        
        % --- Current Predicted/Past Outputs ---
        if h_idx == 1
            yp1 = yKm1;
            yp2 = yKm2;
        elseif h_idx == 2
            yp1 = yhat(h_idx-1);
            yp2 = yKm1;
        else
            yp1 = yhat(h_idx-1);
            yp2 = yhat(h_idx-2);
        end
        
        % --- CRITICAL: Calculate weights dynamically INSIDE the loop ---
        current_operating_point = [u1, yp1]; 
        
        % Compute distance to fuzzy centers
        distances_k = pdist2(centersXY, current_operating_point);
        mu_k = 1 ./ (distances_k.^fuzziness + 1e-6);
        w_k = mu_k / sum(mu_k); % Size: numRules x 1
        
        % --- Build Regressor ---
        Xe_k = [u1, u2, yp1, yp2, 1]; 
        
        % --- Blend local models ---
        yk_pred = 0;
        for i = 1:numRules
            y_local_rule = Xe_k * theta_local(i, :).';
            yk_pred = yk_pred + w_k(i) * y_local_rule;
        end
        
        % Save prediction for the next horizon steps
        yhat(h_idx) = yk_pred;
    end
end

%%
function J = mpc_cost(p, predictor, ref, segLengths, ...
                      yKm1, yKm2, uKm1, state, lamDu) %#ok<DEFNU,INUSD>
% MPC_COST   Receding-horizon cost.
    
    % 1. Force p to be a column vector
    p = p(:);
    
    % 2. Expand segments and predict outputs over horizon
    uSeq = expand_segments(p, segLengths);
    
    % Unpack the model variables nested inside the 'state' struct
    yhat = simulate_horizon_mpc(uSeq, yKm1, yKm2, uKm1, ...
                                state.centersXY, ...
                                state.theta_local, ...
                                state.fuzziness, ...
                                state.numRules);
    
    % Force both reference and prediction to be strict column vectors
    ref = ref(:);
    yhat = yhat(:);
        
    % 4 & 5. Calculate Tracking and Move-Suppression Terms safely
    if length(p) == 1
        uPrev = uKm1;
    else
        uPrev = [uKm1; p(1:end-1)];
    end
    
    % Ensure uPrev matches p's shape exactly
    uPrev = uPrev(:);

    %fprintf("ref   : %dx%d\n", size(ref));
    %fprintf("yhat  : %dx%d\n", size(yhat));
    %fprintf("p     : %dx%d\n", size(p));
    %fprintf("uPrev : %dx%d\n", size(uPrev));
    
    % 6. Total Cost (forced to scalar via 'all' sum)
    J = sum((ref - yhat).^2, 'all') + lamDu * sum((p - uPrev).^2, 'all');
end

function pOpt = solve_mpc_step(predictor, ref, segLengths, ...
                               yKm1, yKm2, uKm1, state, pWarm, cfg, ...
                               uMin, uMax) %#ok<DEFNU,INUSD>
% SOLVE_MPC_STEP   One MPC sub-problem. Return p* clipped to bounds.
    %%%%%%%%%%%%%%%%%%%%  YOUR CODE STARTS HERE  %%%%%%%%%%%%%%%%%%%%
    
    % 1. Determine the length of the horizon window (S) from the warm start vector
    S = length(pWarm);
    
    % 2. Define the anonymous objective function handle
    costFun = @(p) mpc_cost(p, predictor, ref, segLengths, yKm1, yKm2, uKm1, state, cfg.lamDu);
    
    % 3. Branch optimization logic based on horizon length
    if S == 1
        % For a 1-step horizon, use bounded scalar optimization
        pOpt = fminbnd(costFun, uMin, uMax);
    else
        % For multi-step horizons, use Sequential Quadratic Programming (SQP) via fmincon
        opts = optimoptions('fmincon', 'Display', 'off', ...
                            'Algorithm', 'sqp', 'MaxIterations', 40);
                        
        % Vectorize the lower and upper bounds to match the dimension of S
        lb = uMin * ones(S, 1);
        ub = uMax * ones(S, 1);
        
        % Run the constrained optimizer
        pOpt = fmincon(costFun, pWarm, [], [], [], [], lb, ub, [], opts);
    end
    
    % 4. Hard safety clip to plant limits and enforce a column vector shape
    pOpt = min(max(pOpt, uMin), uMax);
    pOpt = pOpt(:); 
    %%%%%%%%%%%%%%%%%%%%%  YOUR CODE ENDS HERE  %%%%%%%%%%%%%%%%%%%%%
end