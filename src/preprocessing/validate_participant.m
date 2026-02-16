function V = validate_participant(P)
% validate_participant  Run consistency checks on a participant struct.
%
%   V = validate_participant(P)
%
% This function runs a series of validation checks on a single participant
% struct P produced by the preprocessing pipeline. It returns a struct V
% containing:
%
%   participant_id     : participant identifier (copied from P)
%   timeline_ok        : true if timeline structure is consistent
%   door_trials_ok     : true if doorTrials data pass basic checks
%   trust_probes_ok    : true if trust probe data pass basic checks
%   questionnaires_ok  : true if questionnaire data are present/valid
%   demographics_ok    : true if demographics info is present/valid
%   emergency_ok       : true if emergency_trial information is present
%                        and interpreted correctly (soft check, does not
%                        gate inclusion in later steps)
%   alignment_ok       : true if a simple alignment check between door
%                        trials and probes is satisfied
%   block_counts       : 1×3 vector with number of door trials per block
%   n_door_trials      : total number of doorTrials entries
%   n_trust_probes     : total number of trust probes
%   issues             : cell array of diagnostic strings describing
%                        which checks failed (if any)
%
% The individual checks rely on helper functions:
%   - check_timeline
%   - check_door_trials
%   - check_trust_probes
%   - check_questionnaires
%   - check_demographics
%   - check_emergency (local to this file)
%   - compute_block_counts
%   - probe_alignment_report
%
% The resulting V struct is later used in Step 3 (validation summary) and
% Step 4 (filtering participants).

    % ---------------------------------------------------------------------
    % Initialize validation summary struct with defaults
    % ---------------------------------------------------------------------
    V = struct();
    V.participant_id     = P.participant_id;
    V.timeline_ok        = false;
    V.door_trials_ok     = false;
    V.trust_probes_ok    = false;
    V.questionnaires_ok  = false;
    V.demographics_ok    = false;
    V.alignment_ok       = false;
    V.emergency_ok       = false;   % emergency_trial response flag (soft check)
    V.block_counts       = [0 0 0];
    V.n_door_trials      = 0;
    V.n_trust_probes     = 0;
    V.issues             = {};

    % ---------------------------------------------------------------------
    % 1) Timeline consistency
    % ---------------------------------------------------------------------
    [ok, msg] = check_timeline(P);
    V.timeline_ok = ok;
    V.issues      = add_issue(V.issues, ok, msg);

    % ---------------------------------------------------------------------
    % 2) Door trials (presence and basic structure)
    % ---------------------------------------------------------------------
    [ok, msg] = check_door_trials(P);
    V.door_trials_ok = ok;
    V.issues         = add_issue(V.issues, ok, msg);
    V.n_door_trials  = numel(P.doorTrials);

    % ---------------------------------------------------------------------
    % 3) Trust probes (presence and basic structure)
    % ---------------------------------------------------------------------
    [ok, msg, nProbes] = check_trust_probes(P);
    V.trust_probes_ok = ok;
    V.issues          = add_issue(V.issues, ok, msg);
    V.n_trust_probes  = nProbes;

    % ---------------------------------------------------------------------
    % 4) Questionnaires (presence and basic parsing)
    % ---------------------------------------------------------------------
    [ok, msg] = check_questionnaires(P);
    V.questionnaires_ok = ok;
    V.issues            = add_issue(V.issues, ok, msg);

    % ---------------------------------------------------------------------
    % 5) Demographics (presence and basic parsing)
    % ---------------------------------------------------------------------
    [ok, msg] = check_demographics(P);
    V.demographics_ok = ok;
    V.issues          = add_issue(V.issues, ok, msg);

    % ---------------------------------------------------------------------
    % 6) Emergency trial (soft check, does not gate inclusion)
    % ---------------------------------------------------------------------
    [ok, msg] = check_emergency(P);
    V.emergency_ok = ok;
    V.issues       = add_issue(V.issues, ok, msg);

    % ---------------------------------------------------------------------
    % 7) Block counts and simple probe–door alignment
    % ---------------------------------------------------------------------
    [counts, okCounts] = compute_block_counts(P);
    V.block_counts = counts;

    [alignOk, alignMsg] = probe_alignment_report(P);

    % Alignment is only considered meaningful if:
    %   - block counts are defined,
    %   - door trials and trust probes passed their own checks.
    V.alignment_ok = okCounts && alignOk && V.door_trials_ok && V.trust_probes_ok;

    V.issues = add_issue(V.issues, V.alignment_ok, alignMsg);
end

% -------------------------------------------------------------------------
% Local helper: add an issue message if a check failed
% -------------------------------------------------------------------------
function L = add_issue(L, ok, msg)
    % ADD_ISSUE  Append msg to issues list if ok == false and msg is non-empty.
    if ~ok && strlength(string(msg)) > 0
        L{end+1} = char(msg); %#ok<AGROW>
    end
end

% -------------------------------------------------------------------------
% Local helper: check presence and validity of emergency_trial response
% -------------------------------------------------------------------------
function [ok, msg] = check_emergency(P)
% check_emergency  Validate the optional emergency_trial response.
%
%   [ok, msg] = check_emergency(P)
%
% Checks whether P.emergency exists and whether it carries a valid parsed
% response ("self" or "robot"). This is a *soft* validation:
%
%   - ok = false  → an issue is recorded in the report
%   - BUT this flag is NOT used as part of the base inclusion mask in
%     Step 4; participants without a valid emergency response are still
%     kept. Downstream, the model can fall back to a default self-
%     confidence value if needed.
%
% Inputs:
%   P - Participant struct, typically containing P.emergency with fields:
%         has_response : logical, true if a valid choice was parsed
%         choice       : string, expected "self" or "robot" if valid
%
% Outputs:
%   ok  - true if emergency field exists and holds a valid choice.
%   msg - diagnostic string (non-empty if ok is false).

    ok  = true;
    msg = "";

    % P.emergency must exist.
    if ~isfield(P, "emergency") || isempty(P.emergency)
        ok  = false;
        msg = "No 'emergency' field found in participant struct (missing emergency_trial info).";
        return;
    end

    E = P.emergency;

    % Check for presence of required fields.
    if ~isfield(E, "has_response") || ~isfield(E, "choice")
        ok  = false;
        msg = "Emergency struct missing has_response/choice fields.";
        return;
    end

    % If we have the struct but no valid response was parsed.
    if ~E.has_response
        ok  = false;
        msg = sprintf("No valid emergency_trial response (choice='%s').", string(E.choice));
        return;
    end

    % Choice should be either "self" or "robot".
    c = lower(string(E.choice));
    if ~(c == "self" || c == "robot")
        ok  = false;
        msg = sprintf("Unexpected emergency_trial choice '%s' (expected 'self' or 'robot').", c);
        return;
    end

    % All checks passed.
    ok  = true;
    msg = "";
end
