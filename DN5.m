clear
close all
clc

%% reset System

pendulum_process_obf([], 0); 


%% Random Staircase Reference

Ts = 0.05;
T_sim = 100;
N = ceil(T_sim/Ts);
t_vec = (0:N-1)'*Ts;

TimeOfPeriod = 2;       % Period of alternation (seconds)
uBaseValue = 1.5;       % Baseline value
amplitude = 0.5;        % How much it steps up/down from baseline

u_vec = zeros(N,1);
numSteps = ceil(t_vec(end) / TimeOfPeriod);

for k = 1:numSteps
    idx = (t_vec >= (k-1)*TimeOfPeriod) & (t_vec < k*TimeOfPeriod);
    stepNoise = (2*rand - 1) * amplitude;
    u_vec(idx) = uBaseValue + stepNoise;
end

% --- ADD SINE WAVE AT THE END HERE ---
T_sine = 40;            % Duration of the sine wave section in seconds
N_sine = ceil(T_sine/Ts);

% Create the extended time vector starting right after the random steps
t_sine = t_vec(end) + Ts + (0:N_sine-1)'*Ts; 

% Sine wave: Amplitude 1, Freq 0.8 Hz, centered around 1
f_sine = 0.2;
u_sine = 1.5 + 0.7 * sin(2 * pi * f_sine * t_sine);

% Append both vectors together
t_vec = [t_vec; t_sine];
u_vec = [u_vec; u_sine];

% Optional: Plot to see the result
plot(t_vec, u_vec);
grid on;
xlabel('Time (s)');
ylabel('u');
title('Random Steps followed by 0.8 Hz Sine Wave');


disp('Začetek simulacije...');
y_vec = pendulum_process_obf(u_vec, Ts);
disp('Simulacija končana.');
refY = y_vec(:,1);


% Limits (degrees)
minAngle = 30;
maxAngle = 60;

% Step duration limits (seconds)
minHold = 1.0;
maxHold = 3.0;


%% Plot
figure
stairs(t_vec,rad2deg(refY),'LineWidth',1.5)
grid on
xlabel('Time [s]')
ylabel('Reference [deg]')
title('Ref signal r()')

%% začetek lab vaje 5
fuzziness = 2;

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

S = 3;                        % Control Horizon (1 decision parameter)
segLengths = [20,20,20];               % A single segment of length 6 (sums to H)
H = sum(segLengths);
cfg.lamDu = 0.5;

% Actuator saturation limits (Adjust these to match your actual pendulum bounds)
uMin = 0.0;                 
uMax = 3.0;                 

% 3. Allocate memory for closed-loop history
u_cl = zeros(N, 1);         % Closed-loop control history
y_cl = zeros(N, 1);         % Closed-loop system output history

% Initialize the system with the first couple of true measurements to kickstart NARX lag
y_cl(1) = refY(1);
y_cl(2) = refY(2);
u_cl(1) = u_vec(1);
u_cl(2) = u_vec(2);

% Warm-start initialization for the optimizer vector p (size S x 1)
pWarm = zeros(S, 1);

disp('Starting MPC Closed-Loop Simulation...');

N = size(t_vec,1)
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
    if mod(k, N/10) == 0
        fprintf('Progress: %.1f%%\n', (k/N)*100);
    end
end

disp('Simulation finished! Plotting results...');
%% Plot MPC Results

figure('Name','MPC Closed Loop Results','NumberTitle','off');

%----------------------------------------------------------
% Output Tracking
%----------------------------------------------------------
subplot(2,1,1)
plot(t_vec, rad2deg(refY),'r--','LineWidth',1.5); hold on;
plot(t_vec, rad2deg(y_cl),'b','LineWidth',1.5);
grid on;
xlabel('Time [s]')
ylabel('Angle [deg]')
title('Reference Tracking')
legend('True Output - Reference','MPC Output','Location','best')

%----------------------------------------------------------
% Control Input
%----------------------------------------------------------
subplot(2,1,2)
plot(t_vec,u_vec,'r--','LineWidth',1.5); hold on;
plot(t_vec,u_cl,'b','LineWidth',1.5); 

grid on
xlabel('Time [s]')
ylabel('Control Input')
title('MPC Control Signal')
legend('True Input','MPC Input','Location','best')



%%

% 1. Calculate the RMSE values
RMSE_y = sqrt(mean((rad2deg(y_cl) - rad2deg(refY)).^2));
RMSE_u = sqrt(mean((u_cl - u_vec).^2)); % Assuming u_vec is the reference for u
MSE_y_rad = mean((rad2deg(y_cl) - rad2deg(refY)).^2);

% 2. Print the results to the Command Window
fprintf('--- Simulation Evaluation Results ---\n');
fprintf('Output Tracking RMSE (y_cl vs refY): %.4f\n', RMSE_y);
fprintf('Control Input RMSE   (u_cl vs u_vec): %.4f\n', RMSE_u);
fprintf('Output Tracking MSE (y_cl vs refY): %.4f\n', MSE_y_rad);


fprintf('-------------------------------------\n');
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
end