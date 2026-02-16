function print_step1_diagnostics(T)
% print_step1_diagnostics  Basic diagnostics for Step 1 preprocessing.
%
%   print_step1_diagnostics(T)
%
% This helper prints a concise summary of the normalized events table T
% after Step 1 of the preprocessing pipeline. It reports:
%   - The set of unique event_type values present in the data.
%   - The count of rows for each event_type.
%   - A small sample of door_trial rows (if any), showing selected columns.
%
% Inputs:
%   T - Event-level table after normalization and projection of common
%       fields. Expected to contain at least 'event_type', and for
%       door_trial events, the columns listed under 'cols' below.

    fprintf('\n=== Step 1 Diagnostics ===\n');

    % ---------------------------------------------------------------------
    % 1) Unique event_type values
    % ---------------------------------------------------------------------
    if ~ismember('event_type', T.Properties.VariableNames)
        warning('No event_type column found.');
        evu = strings(0,1);
    else
        evu = unique(string(T.event_type));
    end

    fprintf('\nUnique event_type values (%d):\n', numel(evu));
    disp(evu);

    % ---------------------------------------------------------------------
    % 2) Counts per event_type
    % ---------------------------------------------------------------------
    fprintf('\nCounts per event_type:\n');
    if ~isempty(evu)
        for i = 1:numel(evu)
            % Count rows whose event_type matches the i-th unique label.
            c = sum(T.event_type == evu(i));
            fprintf('  %-22s : %d\n', evu(i), c);
        end
    end

    % ---------------------------------------------------------------------
    % 3) Sample of door_trial rows (selected columns)
    % ---------------------------------------------------------------------
    if any(T.event_type == "door_trial")
        dmask = (T.event_type == "door_trial");
        D = T(dmask, :);
        n = min(3, height(D));

        fprintf('\nFirst %d door_trial rows (selected columns):\n', n);

        % Restrict display to a subset of columns that are most useful
        % for quick inspection of door trials.
        cols = intersect( ...
            ["block_index","trial_index","suggestion","choice", ...
             "followed","correct","reaction_time_s","timed_out", ...
             "risk_key","risk_value","decision_timeout_ms_used"], ...
            string(D.Properties.VariableNames));

        disp(D(1:n, cols));
    else
        fprintf('\nNo door_trial events found.\n');
    end

    fprintf('==========================\n\n');
end
