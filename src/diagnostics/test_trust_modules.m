% ExperimentalDataProcessing/test_trust_modules.m
%
% TEST_TRUST_MODULES  Basic smoke test for core trust model components.
%
% This script performs a simple end-to-end check of the trust model modules
% using a synthetic (dummy) participant and a small, hand-crafted event
% sequence. It is intended to verify that:
%   - trust_init_state can be called with a minimal parameter/participant
%     struct.
%   - trust_step can process different event types without error.
%   - The resulting trust trajectory is sensible enough for quick visual
%     inspection.
%
% The numerical values used here are not intended to be realistic or
% calibrated; they are only chosen to exercise the code paths.

clear; clc;
addpath("src");

fprintf("=== Testing trust modules with a dummy participant ===\n");

%% 1) Dummy parameters (for code execution, not for inference)
params = struct();

% Dispositional trust (static anchor)
params.disp = struct();
params.disp.tau0 = 0.6;    % dispositional trust

% Reputation component
params.rep = struct();
params.rep.tau0       = 0.2;   % initial reputation component
params.rep.lambda_rep = 0.01;  % reputation decay rate

% Situational trust component (simple linear form here)
params.sit = struct();
params.sit.beta0 = 0.0;   % situational offset
params.sit.beta1 = -0.1;  % situational slope

% Latent trust dynamics
params.lat = struct();
params.lat.lambda_lat = 0.002; % latent drift rate

% Personal experience update (success/failure bumps)
params.exp = struct();
params.exp.gain_succ =  0.05;  % success bump
params.exp.loss_fail = -0.08;  % failure bump

% Relative contribution of rep/sit in trust_step (as used in this test)
params.gamma_rep = 0.5;
params.gamma_sit = 0.3;

%% 2) Dummy participant struct
P = struct();
P.participant_id = "TEST";
P.set_id         = "SetX";
% Other participant fields are not used in this test setup.

%% 3) Initialize trust state
state = trust_init_state(params, P);
disp("Initial state:");
disp(state);

%% 4) Build a toy event sequence
% Event schedule:
%   t = 0   : "q40pre"  (measurement only, no HRI update)
%   t = 10  : door trial, success, followed, risk = 0.5
%   t = 25  : door trial, failure, followed, risk = 0.8
%   t = 40  : "q40post" (measurement only, no HRI update)
%
% The fields 'risk_value', 'outcome', and 'followed' follow the conventions
% of the main trust model (when type == "door").

events = struct([]);

% Event 1: pre-questionnaire (no HRI, just a measurement time)
events(1).t          = 0;            % seconds
events(1).type       = "q40pre";     % not "door", so no HRI update
events(1).risk_value = NaN;          % unused for this type
events(1).outcome    = NaN;
events(1).followed   = NaN;

% Event 2: door trial, success, followed
events(2).t          = 10;
events(2).type       = "door";
events(2).risk_value = 0.5;
events(2).outcome    = 1;           % success
events(2).followed   = 1;           % followed recommendation

% Event 3: door trial, failure, followed
events(3).t          = 25;
events(3).type       = "door";
events(3).risk_value = 0.8;
events(3).outcome    = 0;           % failure
events(3).followed   = 1;

% Event 4: post-questionnaire (no HRI update)
events(4).t          = 40;
events(4).type       = "q40post";
events(4).risk_value = NaN;
events(4).outcome    = NaN;
events(4).followed   = NaN;

%% 5) Step through events and record trajectory
tau_traj = zeros(numel(events),1);
t_traj   = zeros(numel(events),1);

for k = 1:numel(events)
    state = trust_step(state, events(k), params);
    tau_traj(k) = state.tau;
    t_traj(k)   = state.t;
    fprintf("After event %d (%s @ t=%.1f): tau = %.3f\n", ...
        k, events(k).type, events(k).t, state.tau);
end

%% 6) Quick visual check (optional)
figure;
plot(t_traj, tau_traj, '-o');
xlabel('Time (s)');
ylabel('Trust \tau');
title('Dummy trust trajectory (test dynamics)');
grid on;
