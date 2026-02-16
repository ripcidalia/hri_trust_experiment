function met = compute_weighted_metrics(r, w)
% compute_weighted_metrics  Compute weighted residual metrics used throughout the analysis pipeline.
%
% This utility computes a consistent set of error and bias metrics from a vector
% of residuals and corresponding positive weights. It is used across Steps A2–A6
% to ensure that model and baseline evaluations share the exact same metric
% definitions (including filtering rules for invalid samples).
%
% Inputs
%   r (numeric vector)
%       Residuals, typically defined as r = y_obs - y_hat.
%
%   w (numeric vector)
%       Positive weights for each residual. Samples with non-finite weights or
%       w <= 0 are excluded from all metrics.
%
% Outputs
%   met (struct) with fields:
%       N     (double)  Number of valid residual samples after filtering.
%       wRMSE (double)  Weighted root-mean-square error:
%                         sqrt( sum(w .* r.^2) / sum(w) )
%       wMAE  (double)  Weighted mean absolute error:
%                         sum(w .* abs(r)) / sum(w)
%       bias  (double)  Unweighted mean residual (raw bias): mean(r)
%       wBias (double)  Weighted mean residual (weighted bias):
%                         sum(w .* r) / sum(w)
%
% Notes
%   - Filtering: only samples where r and w are finite and w > 0 are retained.
%   - If no valid samples remain (or sum(w) <= 0), all metric fields are NaN.

    r = r(:);
    w = w(:);

    % Keep only finite residuals with strictly positive finite weights.
    ok = isfinite(r) & isfinite(w) & (w > 0);
    r = r(ok);
    w = w(ok);

    met = struct();
    met.N = numel(r);

    % Guard against empty inputs or degenerate weight sums.
    if met.N == 0 || sum(w) <= 0
        met.wRMSE = NaN;
        met.wMAE  = NaN;
        met.bias  = NaN;
        met.wBias = NaN;
        return;
    end

    % Weighted error metrics (normalized by sum(w)).
    met.wRMSE = sqrt( sum(w .* (r.^2)) / sum(w) );
    met.wMAE  =        sum(w .* abs(r)) / sum(w);

    % Bias metrics: raw (unweighted) and weighted.
    met.bias  = mean(r);
    met.wBias = sum(w .* r) / sum(w);
end
