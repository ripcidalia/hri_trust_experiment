function stepM1_ols_calibration(cleanMatPath)
% stepM1_ols_calibration  Global OLS calibration mappings for questionnaire anchors and probes.
%
%   stepM1_ols_calibration(cleanMatPath)
%
% This Step M1 script:
%   1) Loads cleaned participant data containing questionnaire anchors and probe samples.
%   2) Extracts, for each participant, the 40-item questionnaire scores at pre/post and
%      their corresponding 14-item–equivalent scores derived from the same 40-item responses.
%   3) Extracts, for each participant, the trust probe values recorded after the 40-pre
%      and 40-post questionnaires.
%   4) Fits global (pooled across all participants and pre/post anchors) OLS mappings:
%         - 14-equivalent → 40-item scale:
%               Q40 ≈ a14 * Q14_eq + b14
%         - probe value → 40-item scale:
%               Q40 ≈ a1  * P      + b1
%   5) Saves calibration results to:
%         derived/measurement_stepM1_calibration.mat
%
% Inputs:
%   cleanMatPath : (optional) Path to a MAT file containing variable
%                  'participants_clean'.
%                  Default: "derived/participants_time_stepT1.mat".
%
% Outputs (file):
%   The MAT file derived/measurement_stepM1_calibration.mat containing struct:
%
%   calib : struct with fields
%       .a14  - global OLS slope for Q14_eq → Q40 mapping
%       .b14  - global OLS intercept for Q14_eq → Q40 mapping
%       .a1   - global OLS slope for probe → Q40 mapping
%       .b1   - global OLS intercept for probe → Q40 mapping
%       .ids  - N×1 string array of participant IDs
%
% Assumptions:
%   - participants_clean(i) has a field .questionnaires with subfields:
%       t40_pre.total_percent
%       t40_post.total_percent
%       t40_pre.trust14_equiv_total_percent
%       t40_post.trust14_equiv_total_percent
%     all in units of percentage (0..100).
%   - participants_clean(i).trustProbes contains entries with fields:
%       origin, questionnaire_type, value
%     where probes of interest have origin == "after_questionnaire" and
%     questionnaire_type in {"t40_pre","t40_post"}.
%   - There are at least 3 participants to provide a minimally robust pooled fit.

    if nargin < 1 || isempty(cleanMatPath)
        cleanMatPath = "derived/participants_time_stepT1.mat";
    end
    if ~isfile(cleanMatPath)
        error("Clean participants file not found: %s", cleanMatPath);
    end

    % ------------------------------------------------------------
    % 1) Load cleaned participants
    % ------------------------------------------------------------
    S = load(cleanMatPath, "participants_clean");
    if ~isfield(S, "participants_clean")
        error("File %s does not contain 'participants_clean'.", cleanMatPath);
    end
    participants = S.participants_clean;
    N = numel(participants);
    if N < 3
        error("Need at least 3 participants; found %d.", N);
    end

    % ------------------------------------------------------------
    % 2) Extract anchor pairs and probe samples at pre/post
    % ------------------------------------------------------------
    % For each participant we extract:
    %   Q40_pre, Q40_post : 40-item questionnaire scores (0..100)
    %   Q14_pre, Q14_post : 14-item–equivalent scores derived from the 40-item questionnaire (0..100)
    %   P_pre,  P_post    : probe values recorded after the corresponding 40-item questionnaire (0..100)
    %   ids               : participant identifiers

    Q40_pre   = nan(N,1);
    Q40_post  = nan(N,1);
    Q14_pre   = nan(N,1);
    Q14_post  = nan(N,1);
    P_pre     = nan(N,1);
    P_post    = nan(N,1);
    ids       = strings(N,1);

    for i = 1:N
        Pi     = participants(i);
        ids(i) = string(Pi.participant_id);

        q = Pi.questionnaires;
        Q40_pre(i)  = double(q.t40_pre.total_percent);
        Q40_post(i) = double(q.t40_post.total_percent);

        Q14_pre(i)  = double(q.t40_pre.trust14_equiv_total_percent);
        Q14_post(i) = double(q.t40_post.trust14_equiv_total_percent);

        % Extract the probe value sampled immediately after the specified questionnaire
        P_pre(i)  = find_probe_after_q(Pi.trustProbes, "t40_pre");
        P_post(i) = find_probe_after_q(Pi.trustProbes, "t40_post");
    end

    % ------------------------------------------------------------
    % 3) Global OLS mappings (pooled across all participants and anchors)
    % ------------------------------------------------------------

    % Global 14-equivalent → 40 mapping using both pre and post anchors
    Xall_14 = [Q14_pre; Q14_post];
    Yall_14 = [Q40_pre; Q40_post];
    Aall_14 = [Xall_14, ones(numel(Xall_14),1)];
    theta_glob_14 = Aall_14 \ Yall_14;

    a14 = theta_glob_14(1);
    b14 = theta_glob_14(2);

    % Global probe → 40 mapping using both pre and post probe-anchor pairs
    Xall_1        = [P_pre; P_post];
    Yall_1        = [Q40_pre; Q40_post];
    Aall_1        = [Xall_1, ones(numel(Xall_1),1)];
    theta_glob_1  = Aall_1 \ Yall_1;

    a1          = theta_glob_1(1);
    b1          = theta_glob_1(2);

    % ------------------------------------------------------------
    % 4) Pack results and save to disk
    % ------------------------------------------------------------
    calib = struct();
    calib.a14        = a14;
    calib.b14        = b14;
    calib.a1         = a1;
    calib.b1         = b1;
    calib.ids        = ids;

    if ~isfolder("derived")
        mkdir("derived");
    end

    outPath = "derived/measurement_stepM1_calibration.mat";
    save(outPath, "calib", "-v7.3");

    fprintf('[Step M1] Global OLS mappings completed on %d participants.\n', N);
    fprintf('          a_1 = %.4f, b_1 = %.4f, a14 = %.4f, b14 = %.4f\n', ...
            a1, b1, a14, b14);
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
%   - questionnaire_type equals the requested qtype (e.g., "t40_pre", "t40_post").
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
