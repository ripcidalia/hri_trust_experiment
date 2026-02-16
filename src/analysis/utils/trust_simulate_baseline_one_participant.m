function sim = trust_simulate_baseline_one_participant(method, P, dt, weights, baselineParams, cfg)
% trust_simulate_baseline_one_participant  Simulate baseline trust trajectories aligned to the SIMPLE-mode pipeline.
%
% This function provides baseline predictions on the same time grid and with the
% same measurement alignment used by trust_simulate_or_predict_one_participant
% (SIMPLE mode). It relies on get_cached_time_grid_and_events(P, dt) to obtain a
% cached time grid, door events, and an aligned measurement list. The output is
% constructed such that:
%   - sim.measurements is the aligned measurement struct array
%   - sim.y_hat has the same length as sim.measurements
%   - sim.y_hat(m) corresponds 1:1 to sim.measurements(m).y (same ordering)
%
% Supported baselines
%   "const_dispositional"
%       Constant trust equal to the participant's dispositional trust (if
%       available), with fallback to baselineParams.globalConstTrain.
%
%   "const_global_train_mean"
%       Constant trust equal to the global weighted mean computed on TRAIN
%       (passed in baselineParams.globalConstTrain).
%
%   "const_oracle_participant_mean"
%       Per-participant "oracle" constant equal to the participant's own weighted
%       mean of aligned measurements within the current split. Weights are taken
%       from weight_for_kind(kind, weights). Falls back to globalConstTrain.
%
%   "bump_asymmetric"
%       Literature-style bump+saturation baseline in discrete time with global
%       saturation bounds (T in [0,1]):
%           T(k+1) = clip_[0,1]( T(k) + delta_plus*1{success at k}
%                                     - delta_minus*1{failure at k} )
%       Door outcome is mapped to u in {+1,0,-1}. No severity levels are modeled.
%
%   "bump_symmetric"
%       Symmetric bump update at door events (with clipping):
%           T <- clip_[0,1]( T + delta*u ),  u in {+1,0,-1}
%
%   "optimo_lite"
%       OPTIMo-lite Bayesian grid filter baseline:
%         - Prediction updates at door events driven by performance (success/failure)
%         - Probe likelihood updates are ENABLED for filtering, but y_hat at probe
%           times is recorded PRE-UPDATE (one-step-ahead scoring).
%         - Excludes behavioral/intervention evidence (follow/override is NOT used)
%
%   "optimo_lite_outcome_only"
%       OPTIMo-lite ablation (performance-only):
%         - Prediction updates at door events driven by performance
%         - Probe likelihood updates are DISABLED
%         - y_hat at probe times is recorded (equivalently "pre-update", since no update)
%
% Leakage policy for OPTIMo methods:
%   - At any grid index k that contains one or more measurements, y_hat for those
%     measurements is set to the current predictive/filtered mean BEFORE applying
%     the likelihood update(s) for the measurements at k.
%
% Inputs
%   method (string/char)
%       Baseline method identifier (case-insensitive).
%
%   P (struct)
%       Participant struct. Must be compatible with get_cached_time_grid_and_events
%       and with get_dispositional_only(P, fallback).
%
%   dt (double scalar)
%       Time step in seconds for the cached time grid. If omitted/empty, defaults
%       to 1.
%
%   weights (struct)
%       Measurement weights used by the oracle baseline via weight_for_kind.
%
%   baselineParams (struct)
%       Baseline parameter bundle. Expected fields (depending on method):
%         .globalConstTrain (double)  Global TRAIN weighted mean (fallback).
%         .delta_plus (double)        bump_asymmetric success increment (>= 0).
%         .delta_minus (double)       bump_asymmetric failure decrement (>= 0).
%         .delta (double)             bump_symmetric step size (>= 0).
%         .optimo (struct)            OPTIMo-lite fitted params:
%                                      .omega_tb, .omega_tp, .sigma_t
%                                      optionally .omega_td if cfg.optimo.use_trend=true
%
%   cfg (struct)
%       Configuration struct:
%         .clip01 (logical)     If true, clips scalar trust T to [0,1] after updates.
%         .optimo (struct)      Optional OPTIMo-lite settings:
%                               .n_grid, .sigma_f, .sigma0, .use_trend, .boundary
%
% Outputs
%   sim (struct) with fields:
%       .t_grid        (Kx1 double)   Cached time grid (seconds).
%       .doorEvents    (struct array) Door event metadata (from cached helper).
%       .tau_hist      (Kx1 double)   Trust trajectory on the grid.
%       .measurements  (struct array) Aligned measurements (from cached helper).
%       .y_hat         (Mx1 double)   Predicted trust at each measurement time.
%
% Notes
%   - Baseline dynamics are defined at discrete grid steps k.
%   - At each grid index k, any measurements mapped to that index are assigned
%     y_hat = current trust estimate at that time.
%   - Door updates at grid index k use the first door event mapped to k, if any.

    if nargin < 3 || isempty(dt), dt = 1; end
    if nargin < 6 || isempty(cfg), cfg = struct(); end
    if ~isfield(cfg, "clip01"), cfg.clip01 = true; end

    method = string(lower(method));

    % ------------------------------------------------------------------
    % Cached time grid, door events, and aligned measurements
    % ------------------------------------------------------------------
    [t_grid, doors_at_k, doorEvents, meas_at_k, measurements] = ...
        get_cached_time_grid_and_events(P, dt);

    K     = numel(t_grid);
    nMeas = numel(measurements);

    % ------------------------------------------------------------------
    % Baseline parameters (with safe defaults)
    % ------------------------------------------------------------------
    if ~isstruct(baselineParams), baselineParams = struct(); end
    if ~isfield(baselineParams, "globalConstTrain"), baselineParams.globalConstTrain = NaN; end
    globalConstTrain = double(baselineParams.globalConstTrain);

    % ------------------------------------------------------------------
    % Method initialization
    %   - Scalar baselines use scalar state T.
    %   - OPTIMo baselines use internal belief handled by OPTIMo runner.
    % ------------------------------------------------------------------
    T = NaN;            % scalar trust state for non-Bayesian baselines
    delta_plus  = 0;    % bump_asymmetric
    delta_minus = 0;    % bump_asymmetric
    delta       = 0;    % bump_symmetric

    switch method
        case "const_dispositional"
            T = get_dispositional_only(P, globalConstTrain);

        case "const_global_train_mean"
            T = globalConstTrain;

        case "const_oracle_participant_mean"
            T = local_oracle_mean_from_measurements(measurements, weights, globalConstTrain);

        case "bump_asymmetric"
            T = get_dispositional_only(P, globalConstTrain);
            if isfield(baselineParams, "delta_plus"),  delta_plus  = max(double(baselineParams.delta_plus),  0); end
            if isfield(baselineParams, "delta_minus"), delta_minus = max(double(baselineParams.delta_minus), 0); end

        case "bump_symmetric"
            T = get_dispositional_only(P, globalConstTrain);
            if isfield(baselineParams, "delta"), delta = max(double(baselineParams.delta), 0); end

        case {"optimo_lite","optimo_lite_outcome_only"}
            % No scalar initialization.

        otherwise
            error("trust_simulate_baseline_one_participant: unknown baseline method '%s'.", method);
    end

    % Clip scalar initial trust if requested (OPTIMo is bounded by grid anyway)
    if ~startsWith(method, "optimo")
        if cfg.clip01
            T = min(max(T,0),1);
        end
    end

    % ------------------------------------------------------------------
    % Allocate outputs (common interface)
    % ------------------------------------------------------------------
    tau_hist = NaN(K,1);
    y_hat    = NaN(nMeas,1);

    % ------------------------------------------------------------------
    % Run the method (one runner per family)
    % ------------------------------------------------------------------
    switch method
        case {"optimo_lite","optimo_lite_outcome_only"}
            useProbeUpdates = (method == "optimo_lite");
            [tau_hist, y_hat] = local_run_optimo_lite( ...
                P, t_grid, doors_at_k, doorEvents, meas_at_k, measurements, ...
                baselineParams, cfg, useProbeUpdates);

        otherwise
            [tau_hist, y_hat] = local_run_scalar_baseline( ...
                method, T, delta_plus, delta_minus, delta, ...
                doors_at_k, doorEvents, meas_at_k, cfg, K, nMeas);
    end

    % ------------------------------------------------------------------
    % Package outputs (common)
    % ------------------------------------------------------------------
    sim = struct();
    sim.t_grid       = t_grid(:);
    sim.doorEvents   = doorEvents;
    sim.tau_hist     = tau_hist(:);
    sim.measurements = measurements;
    sim.y_hat        = y_hat(:);
end

% =====================================================================
% Scalar baselines (const / bump*) runner
% =====================================================================
function [tau_hist, y_hat] = local_run_scalar_baseline(method, T, delta_plus, delta_minus, delta, doors_at_k, doorEvents, meas_at_k, cfg, K, nMeas)
    tau_hist = NaN(K,1);
    y_hat    = NaN(nMeas,1);

    for k = 1:K
        % Door input u: +1 success, -1 failure, 0 none/unknown
        u = local_door_input_u(doors_at_k{k}, doorEvents);

        % Update
        switch method
            case {"const_dispositional","const_global_train_mean","const_oracle_participant_mean"}
                % No dynamics.

            case "bump_symmetric"
                if u ~= 0
                    T = T + delta * u;
                end

            case "bump_asymmetric"
                if u == +1
                    T = T + delta_plus;
                elseif u == -1
                    T = T - delta_minus;
                end
        end

        if cfg.clip01
            T = min(max(T,0),1);
        end

        tau_hist(k) = T;

        if ~isempty(meas_at_k{k})
            for midx = meas_at_k{k}
                y_hat(midx) = T;
            end
        end
    end
end

function u = local_door_input_u(doorIdxList, doorEvents)
    u = 0;
    if isempty(doorIdxList)
        return;
    end

    d_idx = doorIdxList(1);
    D = doorEvents(d_idx);

    if isfield(D, "outcome") && ~isempty(D.outcome)
        if D.outcome == 1
            u = +1;
        elseif D.outcome == 0
            u = -1;
        else
            u = 0;
        end
    end
end

% =====================================================================
% OPTIMo-lite runner (Bayesian grid filter)
%   - door-driven prediction
%   - optional probe likelihood update
%   - y_hat recorded PRE-UPDATE at measurement indices (leakage fix)
% =====================================================================
function [tau_hist, y_hat] = local_run_optimo_lite(P, t_grid, doors_at_k, doorEvents, meas_at_k, measurements, baselineParams, cfg, useProbeUpdates)
    K     = numel(t_grid);
    nMeas = numel(measurements);

    % ---- Defaults / cfg knobs (safe if cfg.optimo missing) ----
    opt = struct();
    opt.n_grid      = 101;
    opt.sigma_f     = 0.08;       % probe noise (in [0,1] units)
    opt.sigma0      = 0.15;       % initial belief width around dispositional
    opt.use_trend   = false;      % whether to use omega_td*(p_k - p_{k-1})
    opt.boundary    = "truncate"; % truncate+renormalize

    if isfield(cfg, "optimo") && isstruct(cfg.optimo)
        if isfield(cfg.optimo,"n_grid"),    opt.n_grid = cfg.optimo.n_grid; end
        if isfield(cfg.optimo,"sigma_f"),   opt.sigma_f = cfg.optimo.sigma_f; end
        if isfield(cfg.optimo,"sigma0"),    opt.sigma0 = cfg.optimo.sigma0; end
        if isfield(cfg.optimo,"use_trend"), opt.use_trend = logical(cfg.optimo.use_trend); end
        if isfield(cfg.optimo,"boundary"),  opt.boundary = string(cfg.optimo.boundary); end
    end

    % ---- Read fitted OPTIMo-lite parameters ----
    if ~isfield(baselineParams,"optimo") || ~isstruct(baselineParams.optimo)
        error("[optimo_lite] baselineParams.optimo missing. Fit OPTIMo-lite params in Step A5 and pass them in baselineParams.");
    end
    op = baselineParams.optimo;

    req = ["omega_tb","omega_tp","sigma_t"];
    for r = 1:numel(req)
        if ~isfield(op, req(r))
            error("[optimo_lite] Missing optimo parameter '%s' in baselineParams.optimo.", req(r));
        end
    end

    omega_tb = double(op.omega_tb);
    omega_tp = double(op.omega_tp);
    sigma_t  = double(op.sigma_t);

    omega_td = 0;
    if opt.use_trend
        if ~isfield(op,"omega_td")
            error("[optimo_lite] opt.use_trend=true but baselineParams.optimo.omega_td missing.");
        end
        omega_td = double(op.omega_td);
    end

    sigma_f = double(opt.sigma_f);

    % ---- Grid over trust in [0,1] ----
    g = linspace(0,1,opt.n_grid)';  % column

    % ---- Initialize belief around dispositional trust ----
    t0 = get_dispositional_only(P, NaN);
    if ~isfinite(t0), t0 = 0.5; end
    t0 = min(max(t0,0),1);

    b = normpdf(g, t0, max(opt.sigma0, 1e-6));
    b = b / max(sum(b), eps);

    % ---- Outputs ----
    tau_hist = NaN(K,1);
    y_hat    = NaN(nMeas,1);

    % Track performance history (failure indicator p_k)
    p_prev = 0;
    have_p_prev = false;

    for k = 1:K
        % (A) Prediction update at door events (performance-driven)
        idx = doors_at_k{k};
        if ~isempty(idx)
            D = doorEvents(idx(1));

            p_k = local_optimo_pk_from_door(D);

            dp = 0;
            if opt.use_trend
                if have_p_prev
                    dp = (p_k - p_prev);
                else
                    dp = 0;
                end
            end

            A = local_optimo_transition_matrix(g, omega_tb, omega_tp, omega_td, p_k, dp, sigma_t, opt.boundary);
            b = A * b;
            b = b / max(sum(b), eps);

            p_prev = p_k;
            have_p_prev = true;
        end

        % Compute current mean BEFORE any measurement update at k
        t_mean_pre = local_belief_mean(g, b);

        % (B) Leakage fix: assign y_hat for measurements at k using PRE-update mean
        if ~isempty(meas_at_k{k})
            for midx = meas_at_k{k}
                y_hat(midx) = t_mean_pre;
            end
        end

        % (C) Optional probe likelihood update(s) at k (filtering for future)
        if useProbeUpdates && ~isempty(meas_at_k{k})
            b = local_apply_probe_updates(b, g, meas_at_k{k}, measurements, sigma_f);
        end

        % (D) Store trajectory value: posterior mean (after update if enabled)
        tau_hist(k) = local_belief_mean(g, b);
    end
end

function p_k = local_optimo_pk_from_door(D)
    % outcome: 1=success, 0=failure
    if ~isfield(D,"outcome") || isempty(D.outcome) || ~isfinite(double(D.outcome))
        p_k = 0;
        return;
    end
    if D.outcome == 0
        p_k = 1;
    else
        p_k = 0;
    end
end

function m = local_belief_mean(g, b)
    m = sum(g .* b);
    m = min(max(m,0),1);
end

function b = local_apply_probe_updates(b, g, measIdxList, measurements, sigma_f)
    sig = max(double(sigma_f), 1e-8);

    for ii = 1:numel(measIdxList)
        midx = measIdxList(ii);
        y = double(measurements(midx).y);
        if ~isfinite(y)
            continue;
        end
        y = min(max(y,0),1);

        L = normpdf(y, g, sig);   % likelihood as function of latent trust
        b = b .* L;
        b = b / max(sum(b), eps);
    end
end

function A = local_optimo_transition_matrix(g, omega_tb, omega_tp, omega_td, p_k, dp, sigma_t, boundaryMode)
    % Transition:
    %   T_next | T_prev=t ~ N( t + omega_tb + omega_tp*p_k + omega_td*dp, sigma_t^2 )
    % bounded to [0,1] by truncation+renormalization (default).
    n = numel(g);
    A = zeros(n,n);

    sigma_t = max(double(sigma_t), 1e-8);
    boundaryMode = string(boundaryMode);

    for i = 1:n
        mu = g(i) + omega_tb + omega_tp * p_k + omega_td * dp;

        col = normpdf(g, mu, sigma_t);

        if boundaryMode == "truncate"
            s = sum(col);
            if s > 0
                col = col / s;
            else
                [~,j] = min(abs(g - min(max(mu,0),1)));
                col = zeros(n,1);
                col(j) = 1;
            end
        else
            % Default to truncate if unknown
            s = sum(col);
            if s > 0
                col = col / s;
            else
                [~,j] = min(abs(g - min(max(mu,0),1)));
                col = zeros(n,1);
                col(j) = 1;
            end
        end

        A(:,i) = col;
    end
end

% ---------------------------------------------------------------------
% Local helper: oracle weighted mean of aligned measurements for one participant
% ---------------------------------------------------------------------
function mu = local_oracle_mean_from_measurements(measurements, weights, fallback)
    if nargin < 3, fallback = NaN; end
    if isempty(measurements)
        mu = fallback;
        return;
    end

    y = zeros(numel(measurements),1);
    w = zeros(numel(measurements),1);

    for m = 1:numel(measurements)
        y(m) = double(measurements(m).y);
        w(m) = weight_for_kind(string(measurements(m).kind), weights);
    end

    ok = isfinite(y) & isfinite(w) & (w > 0);
    y = y(ok); w = w(ok);

    if isempty(y) || sum(w) <= 0
        mu = fallback;
        return;
    end

    mu = compute_weighted_metrics(y, w).wBias;
    if ~isfinite(mu), mu = fallback; end
end
