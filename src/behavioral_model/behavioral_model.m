function [behavior, p] = behavioral_model(state, params)
% behavioral_model  Predict binary human action given current trust state.
%
%   [behavior, p] = behavioral_model(state)
%
% This function runs an instance of the probabilistic behavioral model
% for a given trust state and parameters of the probabilistic transition.
% It predicts the human action (follow/not follow) as a Bernoulli random
% variable with probability p. The decision boundary is modeled as a smooth
% continuous, sigmoid function of the difference between the trust state
% and the decision threshold:
%
%       p = 1/(1 + exp(-k*(tau-self_confidence))).
%
% Inputs:
%   state     Current trust state with (relevant) fields:
%               .tau         Current total trust in [0,1]
%               .sc          Self-confidence (scalar in [0,1])
%
%   params    Parameters for models.
%
% Output:
%   behavior  Predicted binary action: 1 = followed, 0 = not followed
%   p         Probability of following recommendation

    if nargin < 2 || isempty(params)
        error("behavioral_model: no behavioral parameters specified for coupled mode. Please provide as input.");
    end

    if ~isfield(state, "tau") || isempty(state.tau)
        error("behavioral_model: no trust value specified.");
    else
        tau = state.tau;
    end

    if ~isfield(state, "sc") || isempty(state.sc)
        warning("behavioral_model: no self-confidence value specified. Proceeding with default 0.5");
        self_confidence = 0.5;
    else
        self_confidence = state.sc;
    end

    % Initialize parameters
    
    if isfield(params, "tau_flag")
        tau_flag = params.tau_flag;
    else
        tau_flag = 0;
    end
    
    if isfield(params, "m1_flag")
        m1_flag = params.m1_flag;
    else
        m1_flag = 0;
    end
    
    if isfield(params, "m2_flag")
        m2_flag = params.m2_flag;
    else
        m2_flag = 0;
    end

    flags = [tau_flag, m1_flag, m2_flag];

    if ~any(flags)
        error("behavioral_model: no model flag specified in params.");
    end
    
    if sum(flags) > 1
        error("behavioral_model: more than one model flagged in params.");
    end
    
    if isfield(params, "k_m1")
        k_m1 = params.k_m1;
    else
        k_m1 = -1;
    end
    
    if isfield(params, "k_m2")
        k_m2 = params.k_m2;
    else
        k_m2 = -1;
    end

    if isfield(params, "beta")
        beta = params.beta;
    else
        beta = -1;
    end

    if isfield(params, "eps")
        eps = params.eps;
    else
        eps = -1;
    end

    % Model 0 (direct trust-as-probability):
    %   p_follow = clamp(tau_decision, 0, 1)
    if tau_flag == 1
        p = tau;
    end

    % Model 1 (baseline threshold):
    %   p_follow = sigmoid( k * (tau_decision - self_confidence) )
    if m1_flag == 1 && k_m1 ~= -1
        exp_term = -k_m1*(tau - self_confidence);
        p = 1/(1 + exp(exp_term));
    end

    % Model 2 (offset + lapse):
    %   z = k * tau_decision + beta * (self_confidence - 0.5)
    %   p*_follow = sigmoid(z)
    %   p_follow  = (1-eps)*p*_follow + eps*0.5
    if m2_flag == 1 && k_m2 ~= -1 && beta ~= -1 && eps ~= -1
        exp_term = k_m2*tau + beta*(self_confidence - 0.5);
        p_star = 1/(1 + exp(-exp_term));
        p = (1 - eps)*p_star + eps*0.5;
    end

    if rand() < p
        behavior = 1;
    else
        behavior = 0;
    end