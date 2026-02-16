function Tout = diag_behavior_thresholds(run_id, splitName, varargin)
% diag_behavior_thresholds  Print per-participant decision-threshold diagnostics from A7 dataset.
%
% For each participant and scope (global + block1..3), compute:
%   1) min tau_decision where followed==1
%   2) max tau_decision where followed==0
%   3) self_confidence
%   4) gap = min_follow - max_override
%
% IMPORTANT:
%   Uses tau_decision from A7, which is pre-door trust:
%     tau_decision = tau_hist(k_grid-1)
%   This avoids using post-update trust at the door event time.
%
% Usage:
%   diag_behavior_thresholds("RUN_001", "valid");
%   T = diag_behavior_thresholds("RUN_001", "train", "Print", false);

p = inputParser;
p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
p.addParameter("Print", true, @(x) islogical(x) && isscalar(x));
p.parse(varargin{:});
args = p.Results;

run_id = string(run_id);
splitName = lower(string(splitName));
if ~(splitName=="train" || splitName=="valid")
    error("splitName must be 'train' or 'valid'. Got: %s", splitName);
end

% Locate A7 output
outDir = string(args.OutDir);
if strlength(outDir)==0
    outDir = fullfile("derived","analysis_runs",run_id,"stepA7_behavior_dataset");
end
matPath = fullfile(outDir, sprintf("behavior_dataset_%s.mat", splitName));
if ~isfile(matPath)
    error("A7 dataset not found: %s", matPath);
end

S = load(matPath, "T");
if ~isfield(S, "T")
    error("MAT file missing variable T: %s", matPath);
end
T = S.T;

% Validity mask: require finite tau_decision and a valid followed label
if ismember("is_valid_label", T.Properties.VariableNames)
    isValidLabel = T.is_valid_label;
else
    isValidLabel = isfinite(T.followed) & (T.followed==0 | T.followed==1);
end
isValid = isValidLabel & isfinite(T.tau_decision);

% Participant list
pids = unique(T.participant_id);
nP = numel(pids);

% Preallocate output rows: 4 per participant
nRows = nP * 4;
participant_id    = strings(nRows,1);
scope             = strings(nRows,1);
min_tau_follow    = NaN(nRows,1);
max_tau_override  = NaN(nRows,1);
gap               = NaN(nRows,1);
self_conf         = NaN(nRows,1);

row = 0;

for i = 1:nP
    pid = pids(i);

    maskPid = (T.participant_id == pid) & isValid;

    % Self-confidence: constant per participant (take first finite)
    scv = T.self_confidence(T.participant_id==pid);
    scv = scv(isfinite(scv));
    sc = NaN;
    if ~isempty(scv), sc = scv(1); end

    % ---- Global ----
    row = row + 1;
    participant_id(row) = pid;
    scope(row) = "global";
    self_conf(row) = sc;

    maskFollow = maskPid & (T.followed==1);
    maskOver   = maskPid & (T.followed==0);

    if any(maskFollow)
        min_tau_follow(row) = min(T.tau_decision(maskFollow));
    end
    if any(maskOver)
        max_tau_override(row) = max(T.tau_decision(maskOver));
    end
    if isfinite(min_tau_follow(row)) && isfinite(max_tau_override(row))
        gap(row) = min_tau_follow(row) - max_tau_override(row);
    end

    % ---- Per block ----
    for b = 1:3
        row = row + 1;
        participant_id(row) = pid;
        scope(row) = "block" + string(b);
        self_conf(row) = sc;

        maskB = maskPid & (T.block_index == b);
        maskFollowB = maskB & (T.followed==1);
        maskOverB   = maskB & (T.followed==0);

        if any(maskFollowB)
            min_tau_follow(row) = min(T.tau_decision(maskFollowB));
        end
        if any(maskOverB)
            max_tau_override(row) = max(T.tau_decision(maskOverB));
        end
        if isfinite(min_tau_follow(row)) && isfinite(max_tau_override(row))
            gap(row) = min_tau_follow(row) - max_tau_override(row);
        end
    end
end

Tout = table(participant_id, scope, min_tau_follow, max_tau_override, gap, self_conf);

% Sort nicely: participant then global->blocks
scopeOrder = categorical(scope, ["global","block1","block2","block3"], "Ordinal", true);
Tout.scope = string(scopeOrder);
% Sort nicely: participant then global->blocks
Tout.scope_sort = categorical(Tout.scope, ...
    ["global","block1","block2","block3"], "Ordinal", true);

Tout = sortrows(Tout, {'participant_id','scope_sort'});
Tout.scope_sort = [];  % remove helper column


if args.Print
    fprintf("\n[A7 thresholds] run_id=%s split=%s\n", run_id, splitName);
    fprintf("Uses pre-door trust tau_decision from A7 (tau_hist(k_grid-1)).\n\n");
    disp(Tout);
end
end
