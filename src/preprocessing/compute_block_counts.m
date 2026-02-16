function [counts, ok] = compute_block_counts(P)
% compute_block_counts  Count door trials per block timeline.
%
%   [counts, ok] = compute_block_counts(P)
%
% This function counts the number of door trials in each block for a given
% participant struct P. It inspects the block-specific timelines:
%   - P.block1_timeline
%   - P.block2_timeline
%   - P.block3_timeline
%
% A door trial is identified by entries whose string label starts with
% "door_". The function returns:
%
%   counts : 1×3 numeric vector with the number of door trials per block.
%            counts(1) corresponds to block 1, counts(2) to block 2, etc.
%
%   ok     : logical flag indicating whether the block counts satisfy a
%            basic sanity condition. In the current implementation this
%            flag is always set to true. It is retained for compatibility
%            with earlier design ideas, where additional checks could be
%            applied (e.g., non-decreasing counts across existing blocks).
%
% Inputs:
%   P - Participant struct that may contain fields:
%         block1_timeline, block2_timeline, block3_timeline
%       Each block*_timeline is expected to be a string array of event
%       labels, such as "door_1_1", "door_1_2", or non-door labels.
%
% Outputs:
%   counts - 1×3 double, number of door entries per block.
%   ok     - logical, true if block counts pass basic sanity checks.
%            (Currently always true.)
%
% Notes:
%   - If a given block*_timeline field is missing or empty, the
%     corresponding count is set to 0.
%   - The definition of "door trial" is purely string-based here (entries
%     starting with "door_"), which is consistent with how timelines are
%     constructed in the preprocessing pipeline.

    % ---------------------------------------------------------------------
    % 1) Collect block timelines in a cell array for convenience
    % ---------------------------------------------------------------------
    seqs = {
        safe_seq(P, "block1_timeline"), ...
        safe_seq(P, "block2_timeline"), ...
        safe_seq(P, "block3_timeline") ...
    };

    counts = zeros(1, 3);

    % ---------------------------------------------------------------------
    % 2) Count door labels ("door_*") in each block's timeline
    % ---------------------------------------------------------------------
    for b = 1:3
        s = seqs{b};
        if isempty(s)
            counts(b) = 0;
            continue;
        end

        % A door trial is any timeline entry whose label starts with "door_".
        isDoor = startsWith(s, "door_");
        counts(b) = sum(isDoor);
    end

    % ---------------------------------------------------------------------
    % 3) Basic sanity flag
    % ---------------------------------------------------------------------
    % In this implementation, we always mark the block counts as OK. The
    % 'ok' output is kept to allow future extension (e.g., checking that
    % later blocks do not have door trials if earlier blocks are empty).
    ok = true;
end

% -------------------------------------------------------------------------
% Local helper: safely get a block timeline from participant struct
% -------------------------------------------------------------------------

function s = safe_seq(P, name)
    % SAFE_SEQ  Return P.(name) if it exists, otherwise an empty string array.
    if isfield(P, name)
        s = P.(name);
    else
        s = strings(0, 1);
    end
end
