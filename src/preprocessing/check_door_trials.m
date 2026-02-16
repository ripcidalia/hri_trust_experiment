function [ok, msg] = check_door_trials(P)
% check_door_trials  Validate the presence and basic completeness of door trials.
%
%   [ok, msg] = check_door_trials(P)
%
% This function verifies that the participant struct contains a non-empty
% set of door trials and that each door trial includes both a non-empty
% suggestion and choice. These fields are required for the trust-update
% logic and cost function.
%
% Inputs:
%   P   - Participant struct produced by preprocessing, expected to contain:
%           P.doorTrials : 1×N struct array, each element representing a
%                          door trial with fields such as:
%                              suggestion   : string
%                              choice       : string
%
% Outputs:
%   ok  - Logical flag indicating whether the participant's door trials
%         meet the minimum completeness requirements.
%   msg - Diagnostic message. Empty if ok is true; otherwise a brief
%         explanation of the detected issue.

    % Extract the doorTrials array. Its absence or emptiness indicates that
    % the participant never encountered door events, which is invalid for
    % the trust-model pipeline.
    D = P.doorTrials;
    if isempty(D)
        ok = false;
        msg = "No door trials found.";
        return;
    end

    % Check for any missing suggestion or choice entries. A door trial must
    % record both fields for the model to determine whether the participant
    % followed the robot's advice and compute corresponding trust dynamics.
    suggEmpty   = arrayfun(@(x) strlength(x.suggestion) == 0, D);
    choiceEmpty = arrayfun(@(x) strlength(x.choice) == 0, D);

    % Identify the first occurrence of missing data, if any.
    missing = find(suggEmpty | choiceEmpty, 1);
    if ~isempty(missing)
        ok = false;
        msg = sprintf("Door trial %d missing suggestion or choice.", missing);
        return;
    end

    % All door trials contain the required fields.
    ok  = true;
    msg = "";
end
