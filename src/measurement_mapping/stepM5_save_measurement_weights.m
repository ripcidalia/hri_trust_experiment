function stepM5_save_measurement_weights(residualVarsPath)
% stepM5_save_measurement_weights  Compute and save unified measurement weights.
%
%   stepM5_save_measurement_weights(residualVarsPath)
%
% Step M5 consolidates measurement weighting parameters used downstream in
% Weighted Least Squares (WLS) trust fitting. It loads residual variance
% estimates from Step M3 and computes relative weights for:
%   - 40-item questionnaire totals (reference instrument)
%   - 14-item questionnaire totals (treated via a fixed proportional rule)
%   - Single trust probes (computed from variance estimates)
%
% The resulting weights struct is saved to disk for use in trust cost
% functions that combine heterogeneous measurement types on a common scale.
%
% Inputs:
%   residualVarsPath : (optional) Path to MAT file containing struct
%                     'residualVars' with fields:
%                         .var_anchor  (anchor-context residual variance)
%                         .var_mid     (mid-block-context residual variance)
%                     Default:
%                         "derived/measurement_step3_residual_variances.mat"
%
% Output (file):
%   Writes "derived/measurement_weights.mat" containing struct weights with fields:
%       .w40         - reference weight for 40-item questionnaire (set to 1)
%       .w14         - weight for 14-item questionnaire (set via fixed proportion)
%       .w_probe     - weight for probe measurements (computed from residual variances)
%       .var_anchor  - residual variance estimate from pre/post anchor context
%       .var_mid     - residual variance estimate from mid-block context
%       .info        - metadata (source file, timestamp, description)
%
% Notes on weight construction:
%   - The 40-item questionnaire is defined as the reference scale (w40 = 1).
%   - The 14-item questionnaire weight is set by a fixed ratio (14/40).
%   - The probe weight is computed from var_anchor and var_mid using the
%     expression implemented below; this yields a weight intended to be
%     relative to the 40-item reference scale.
%
% Assumptions:
%   - residualVarsPath contains 'residualVars' with the expected fields.
%   - This step packages results only; it does not modify participant data.

    % -----------------------------
    % Handle file inputs
    % -----------------------------
    if nargin < 1 || isempty(residualVarsPath)
        residualVarsPath = "derived/measurement_step3_residual_variances.mat";
    end

    if ~isfile(residualVarsPath)
        error("Residual variances file not found: %s", residualVarsPath);
    end

    % -----------------------------
    % Load residual variance estimates
    % -----------------------------
    R = load(residualVarsPath, "residualVars");

    residualVars = R.residualVars;
    var_anchor   = residualVars.var_anchor;
    var_mid      = residualVars.var_mid;
    
    % ----------------------------------------------------------
    % Compute weights
    % ----------------------------------------------------------
    % Define relative weights normalized to the 40-item instrument.
    w40 = 1;  % reference scale
    
    % Fixed proportional weight for 14-item measurements relative to 40-item
    w14 = 14/40;

    % Probe weight computed from residual variance estimates
    w1  = 26/((40*var_anchor)-(14*var_mid));

%     % Optional guard logic can be enabled to bound w1 to conservative ranges
%     % if variance estimates are considered unreliable in low-data regimes.
% 
%     if w1 < 1/80 || w1 > w14 % for extremely small or abnormaly large 
%         w1 = 1/40;           % weights, default to conservative guess
%     end

    % -----------------------------
    % Build merged weights struct
    % -----------------------------
    weights = struct();
    weights.w40        = w40;                    
    weights.w14        = w14;
    weights.w_probe    = w1;

    weights.var_anchor = var_anchor;
    weights.var_mid    = var_mid;

    % Metadata about how these weights were constructed
    weights.info = struct();
    weights.info.residualVars_source = residualVarsPath;
    weights.info.created = char(datetime('now', ...
                                'Format','yyyy-MM-dd HH:mm:ss'));
    weights.info.description = ...
        "Unified measurement weights for 40-item, 14-item, and probe instruments.";

    % -----------------------------
    % Save weights
    % -----------------------------
    if ~isfolder("derived")
        mkdir("derived");
    end

    outPath = "derived/measurement_weights.mat";
    save(outPath, "weights", "-v7.3");

    fprintf("[Step M5] Measurement weights saved to %s\n", outPath);
    fprintf("          w40 = %.3f, w14 = %.3f, w_probe = %.3f\n", ...
        weights.w40, weights.w14, weights.w_probe);
end
