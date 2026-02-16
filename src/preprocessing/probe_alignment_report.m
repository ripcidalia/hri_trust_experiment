function [ok, msg] = probe_alignment_report(P)
% probe_alignment_report  Check alignment between trust probes and door IDs.
%
%   [ok, msg] = probe_alignment_report(P)
%
% This function performs a lightweight consistency check between the
% participant's trust probes and their timeline of door events.
%
% Specifically, it verifies that any probe whose origin is "after_door"
% and that carries valid (block_index, trial_index) information points to
% a door identifier that actually exists in the participant's timeline.
%
% Probes whose origin is "after_questionnaire" are ignored here, because
% they are conceptually linked to questionnaire events rather than door
% trials.
%
% Inputs:
%   P   - Participant struct with fields:
%           P.trustProbes : 1×M struct array with fields:
%                             origin      : "after_door" or "after_questionnaire"
%                             block_index : 1-based block index (if after_door)
%                             trial_index : within-block door order (if after_door)
%           P.timeline    : string array of event labels, including door IDs
%                          such as "door_B_K" (B = block, K = within-block index).
%
% Outputs:
%   ok  - Logical flag indicating whether all checked probes refer to door
%         IDs that are present in the participant timeline.
%   msg - Diagnostic message:
%           "" if ok == true
%           A description of the first misalignment found, or a note that
%           no after_door probes with indices were available for checking.

    % Default outcome: assume alignment is OK unless a violation is found.
    ok  = true;
    msg = "";

    % If there are no trust probes at all, there is nothing to check.
    U = P.trustProbes;
    if isempty(U)
        return;
    end

    % Extract all door IDs from the timeline (e.g., "door_1_1", "door_2_3").
    tl = P.timeline;
    doorIds = tl(startsWith(tl, "door_"));

    % Flag to track whether we actually checked any after_door probes.
    hasAnyChecked = false;

    % ---------------------------------------------------------------------
    % Loop over all probes and verify those that follow a door.
    % ---------------------------------------------------------------------
    for i = 1:numel(U)
        % Skip probes that explicitly follow questionnaires; they are not
        % expected to reference a door ID.
        if isfield(U(i), "origin") && U(i).origin == "after_questionnaire"
            continue;
        end

        % Only attempt alignment check if block_index and trial_index exist.
        if isfield(U(i), "block_index") && isfield(U(i), "trial_index")
            b = U(i).block_index;
            t = U(i).trial_index;
            if ~isnan(b) && ~isnan(t)
                hasAnyChecked = true;

                % Construct expected door ID from block and trial indices.
                probeId = "door_" + string(b) + "_" + string(t);

                % If this ID does not appear anywhere in the timeline, flag
                % the mismatch and bail out on the first occurrence.
                if ~any(doorIds == probeId)
                    ok  = false;
                    msg = sprintf("Probe #%d refers to %s absent from timeline.", i, probeId);
                    return;
                end
            end
        end
    end

    % If we never encountered an after_door probe with usable indices,
    % report that there was simply nothing to check.
    if ~hasAnyChecked
        msg = "No after_door probes with indices to check.";
    end
end
