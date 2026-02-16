function [ok, msg] = check_timeline(P)
% check_timeline  Validate that a participant timeline is non-empty.
%
%   [ok, msg] = check_timeline(P)
%
% This helper performs a minimal validation on the participant's timeline:
% it checks only that the timeline field is non-empty. Chronological
% ordering and timestamp sanity are enforced earlier in the pipeline
% (during Step 1), so no ordering or content checks are performed here.
%
% Inputs:
%   P   - Participant struct produced by the preprocessing pipeline.
%         It is expected to contain a field:
%             P.timeline : string or cell array of timeline labels
%                          (e.g., "door_1_1", "questionnaire40pre", ...).
%
% Outputs:
%   ok  - Logical flag, true if the timeline is non-empty, false otherwise.
%   msg - Diagnostic message. Empty if ok is true; otherwise contains a
%         short description of the issue (e.g., "Timeline is empty.").

    % Extract the timeline from the participant struct. At this stage we
    % assume that P.timeline has already been constructed by
    % build_participant_struct and reflects the full event sequence.
    tl = P.timeline;

    % Basic check: consider the timeline valid if it contains at least one
    % entry; otherwise flag as invalid.
    ok = ~isempty(tl);

    % Initialize diagnostic message and fill only in the failure case.
    msg = "";
    if ~ok
        msg = "Timeline is empty.";
    end
end
