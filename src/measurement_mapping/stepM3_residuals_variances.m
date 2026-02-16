function stepM3_residuals_variances(cleanMatPath)
% stepM3_residuals_variances  Estimate probe-to-questionnaire residual variances via LOPO.
%
%   stepM3_residuals_variances(cleanMatPath)
%
% This Step M3 function estimates two residual variances for single trust
% probe measurements (slider ratings) when linearly mapped to 40-item
% questionnaire percentages. The estimates are obtained via
% leave-one-participant-out (LOPO) cross-validation in two contexts:
%
%   A) Anchor context (pre/post):
%      - Targets: 40-item questionnaire totals at t40_pre and t40_post
%      - Predictors: trust probe values recorded after t40_pre and t40_post
%
%   B) Mid-block context (mid1/mid2):
%      - Targets: mapped mid-block questionnaire totals already expressed on
%                 the 40-item scale (t14_mid1.total_percent_40 and
%                 t14_mid2.total_percent_40, produced in Step M2)
%      - Predictors: trust probe values recorded after t14_mid1 and t14_mid2
%
% For each context, LOPO fits a simple global linear model on the training
% participants:
%       Q40_hat = a1_i * Probe + b1_i
% then evaluates residuals for the held-out participant and pools residuals
% across all participants to compute a sample variance estimate.
%
% Inputs:
%   cleanMatPath : (optional) Path to MAT file containing 'participants_mapped'
%                  (output of Step M2). Default:
%                  "derived/participants_mapped14_stepM2.mat".
%
% Output (file):
%   Writes "derived/measurement_step3_residual_variances.mat" containing struct
%   residualVars with fields:
%       .var_anchor  - pooled sample variance of LOPO residuals for pre/post anchors
%       .var_mid     - pooled sample variance of LOPO residuals for mid-block anchors
%       .n_anchor    - number of non-NaN residuals used for var_anchor
%       .n_mid       - number of non-NaN residuals used for var_mid
%
% Assumptions:
%   - Each participant has questionnaire fields:
%       t40_pre.total_percent, t40_post.total_percent (0..100),
%       t14_mid1.total_percent_40, t14_mid2.total_percent_40 (0..100 or NaN).
%   - Each participant has trustProbes entries with fields:
%       origin, questionnaire_type, value
%     where relevant probes have origin == "after_questionnaire" and
%     questionnaire_type in {"t40_pre","t40_post","t14_mid1","t14_mid2"}.
%   - There are sufficient non-NaN probe/target pairs to estimate variances
%     in each context.

    if nargin < 1 || isempty(cleanMatPath)
        cleanMatPath = "derived/participants_mapped14_stepM2.mat";
    end

    if ~isfile(cleanMatPath)
        error("participants_mapped file not found: %s", cleanMatPath);
    end

    % ------------------------------------------------------------
    % 1) Load participants (with mapped mid-block questionnaire totals)
    % ------------------------------------------------------------
    S = load(cleanMatPath, "participants_mapped");
    participants = S.participants_mapped;
    N = numel(participants);

    % ------------------------------------------------------------
    % 2) Context A: pre/post anchors (t40_pre/t40_post) and their probes
    % ------------------------------------------------------------
    % Extract per-participant targets and predictors:
    %   Q40_pre, Q40_post : 40-item questionnaire totals
    %   P_pre,  P_post    : probe values recorded after those questionnaires

    Q40_pre   = nan(N,1);
    Q40_post  = nan(N,1);
    P_pre     = nan(N,1);
    P_post    = nan(N,1);
    ids       = strings(N,1);

    for i = 1:N
        Pi     = participants(i);
        ids(i) = string(Pi.participant_id);

        q = Pi.questionnaires;
        Q40_pre(i)  = double(q.t40_pre.total_percent);
        Q40_post(i) = double(q.t40_post.total_percent);

        % Probe sampled after the corresponding questionnaire completion
        P_pre(i)  = find_probe_after_q(Pi.trustProbes, "t40_pre");
        P_post(i) = find_probe_after_q(Pi.trustProbes, "t40_post");
    end

    % LOPO residuals for anchor context
    res_pre  = nan(N,1);
    res_post = nan(N,1);

    for i = 1:N
        % Training set: all participants except i
        J = setdiff(1:N, i);

        x_train = [P_pre(J);  P_post(J)];      % probe predictors
        y_train = [Q40_pre(J); Q40_post(J)];   % questionnaire targets

        % Fit linear model on training set: y = a1_i * x + b1_i
        A     = [x_train, ones(numel(x_train),1)];
        theta = A \ y_train;
        a1_i  = theta(1);
        b1_i  = theta(2);

        % Evaluate held-out participant at pre and post
        yhat_pre  = a1_i * P_pre(i)  + b1_i;
        yhat_post = a1_i * P_post(i) + b1_i;

        res_pre(i)  = Q40_pre(i)  - yhat_pre;
        res_post(i) = Q40_post(i) - yhat_post;
    end

    % Pooled sample variance across all non-NaN anchor residuals
    E_anchor = [res_pre; res_post];
    E_anchor = E_anchor(~isnan(E_anchor));
    if numel(E_anchor) < 2
        error("Not enough non-NaN anchor residuals to estimate probe variance.");
    end
    e_bar      = mean(E_anchor);
    n_anchor   = numel(E_anchor);
    var_anchor = sum((E_anchor - e_bar).^2) / (n_anchor - 1);


    % ------------------------------------------------------------
    % 3) Context B: mid-block mapped totals (t14_mid1/t14_mid2) and probes
    % ------------------------------------------------------------
    % Extract per-participant targets and predictors:
    %   Q40_mid1, Q40_mid2 : mid-block questionnaire totals on the 40-item scale
    %   P_mid1,  P_mid2    : probe values recorded after those questionnaires

    Q40_mid1  = nan(N,1);
    Q40_mid2  = nan(N,1);
    P_mid1    = nan(N,1);
    P_mid2    = nan(N,1);
    ids       = strings(N,1);

    for i = 1:N
        Pi     = participants(i);
        ids(i) = string(Pi.participant_id);

        q = Pi.questionnaires;
        Q40_mid1(i) = double(q.t14_mid1.total_percent_40);
        Q40_mid2(i) = double(q.t14_mid2.total_percent_40);

        % Probe sampled after the corresponding mid-block questionnaire completion
        P_mid1(i)  = find_probe_after_q(Pi.trustProbes, "t14_mid1");
        P_mid2(i)  = find_probe_after_q(Pi.trustProbes, "t14_mid2");
    end

    % LOPO residuals for mid-block context
    res_mid1 = nan(N,1);
    res_mid2 = nan(N,1);

    for i = 1:N
        % Training set: all participants except i
        J = setdiff(1:N, i);

        x_train = [P_mid1(J);  P_mid2(J)];      % probe predictors
        y_train = [Q40_mid1(J); Q40_mid2(J)];   % mapped questionnaire targets

        % Fit linear model on training set: y = a1_i * x + b1_i
        A     = [x_train, ones(numel(x_train),1)];
        theta = A \ y_train;
        a1_i  = theta(1);
        b1_i  = theta(2);

        % Evaluate held-out participant at mid1 and mid2
        yhat_mid1 = a1_i * P_mid1(i) + b1_i;
        yhat_mid2 = a1_i * P_mid2(i) + b1_i;

        res_mid1(i) = Q40_mid1(i) - yhat_mid1;
        res_mid2(i) = Q40_mid2(i) - yhat_mid2;
    end

    % Pooled sample variance across all non-NaN mid-block residuals
    E_mid = [res_mid1; res_mid2];
    E_mid = E_mid(~isnan(E_mid));
    if numel(E_mid) < 2
        error("Not enough non-NaN anchor residuals to estimate probe variance.");
    end
    e_bar   = mean(E_mid);
    n_mid   = numel(E_mid);
    var_mid = sum((E_mid - e_bar).^2) / (n_mid - 1);

    % ------------------------------------------------------------
    % 4) Pack results and save to disk
    % ------------------------------------------------------------
    residualVars = struct();
    residualVars.var_anchor = var_anchor;
    residualVars.var_mid    = var_mid;
    residualVars.n_anchor   = n_anchor;
    residualVars.n_mid      = n_mid;

    if ~isfolder("derived")
        mkdir("derived");
    end
    outPath = "derived/measurement_step3_residual_variances.mat";
    save(outPath, "residualVars", "-v7.3");

    fprintf('[Step M3] Residual variances estimation completed on %d participants.\n', N);
    fprintf('          var_anchor = %.4f, var_mid = %.4f\n', ...
            var_anchor, var_mid);
end

% -------------------------------------------------------------------------
% Local helpers
% -------------------------------------------------------------------------

function v = find_probe_after_q(trustProbes, qtype)
% find_probe_after_q  Extract probe value recorded after a given questionnaire.
%
%   v = find_probe_after_q(trustProbes, qtype)
%
% Returns the numeric probe value (0..100) for the first probe entry whose:
%   - origin equals "after_questionnaire", and
%   - questionnaire_type equals the requested qtype (e.g., "t40_pre",
%     "t40_post", "t14_mid1", "t14_mid2").
%
% Returns NaN if no matching probe is found or if the probe value is missing.

    v = NaN;
    for k = 1:numel(trustProbes)
        tp = trustProbes(k);
        if isfield(tp,"origin") && tp.origin == "after_questionnaire" ...
                && isfield(tp,"questionnaire_type") && tp.questionnaire_type == qtype
            if isfield(tp,"value")
                v = double(tp.value);
                return;
            end
        end
    end
end
