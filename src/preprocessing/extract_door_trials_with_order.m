function doorTrials = extract_door_trials_with_order(Tk, door_order_index)
% extract_door_trials_with_order  Door trials with within-block order indices.
%
%   doorTrials = extract_door_trials_with_order(Tk, door_order_index)
%
% This function extracts all door_trial events from a participant-level
% table Tk and constructs a 1×N struct array, similarly to
% extract_door_trials, but with additional structure on the indices:
%
%   - block_index      : 1-based block index (converted from 0-based raw).
%   - trial_index      : within-block door order index (1..N per block),
%                        provided via door_order_index.
%   - trial_index_raw  : original global trial_index from the logs (0-based).
%
% The remaining fields (suggestion, choice, followed, correct, etc.) match
% those in extract_door_trials. The within-block order allows consistent
% labelling of doors as "door_b_k" for block b and within-block index k.
%
% Inputs:
%   Tk               - Table of events for a single participant.
%   door_order_index - Column vector (same height as Tk) giving within-block
%                      door order indices for rows where event_type ==
%                      "door_trial". Non-door rows are typically NaN.
%
% Outputs:
%   doorTrials       - 1×N struct array of door trials with enriched index
%                      information. If there are no door_trial rows,
%                      doorTrials is an empty struct array.
%
% Assumptions:
%   - Tk contains event_type, block_index, trial_index, and all columns
%     accessed below.
%   - door_order_index has already been computed with the same row
%     ordering as Tk.

    % Select only door_trial rows.
    mask = (string(Tk.event_type) == "door_trial");
    D = Tk(mask, :);
    order = door_order_index(mask);

    % If there are no door trials, return an empty struct array.
    if isempty(D)
        doorTrials = struct([]);
        return;
    end

    % Preallocate struct array with all required fields.
    N = height(D);
    doorTrials = repmat(struct( ...
        'block_index', NaN, ...
        'trial_index', NaN, ...           % within-block door order (1-based)
        'trial_index_raw', NaN, ...       % original global trial_index (0-based)
        'suggestion', "", ...
        'choice', "", ...
        'followed', false, ...
        'correct', false, ...
        'timed_out', false, ...
        'reaction_time_s', NaN, ...
        'risk_key', "", ...
        'risk_value', NaN, ...
        'decision_timeout_ms_used', NaN, ...
        'background_src', "", ...
        'door_src', "", ...
        'victim_skin', "", ...
        'door_id', "" ), N, 1);

    % Local helper handles for numeric, string, and logical extraction.
    num  = @(name, i) get_num(D, name, i);
    str  = @(name, i) get_str(D, name, i);
    logi = @(name, i) get_log(D, name, i);

    % Populate each door trial entry.
    for i = 1:N
        b0 = num("block_index", i);
        t0 = num("trial_index", i);

        doorTrials(i).block_index     = b0 + 1;      % 1-based block index
        doorTrials(i).trial_index     = order(i);    % within-block 1..N
        doorTrials(i).trial_index_raw = t0;          % raw global trial index (0-based)

        doorTrials(i).suggestion  = str("suggestion", i);
        doorTrials(i).choice      = str("choice", i);
        doorTrials(i).followed    = logi("followed", i);
        doorTrials(i).correct     = logi("correct", i);
        doorTrials(i).timed_out   = logi("timed_out", i);

        % Reaction time: prefer reaction_time_s, otherwise rt_ms / 1000.
        rts = num("reaction_time_s", i);
        if isnan(rts)
            rtms = num("rt_ms", i);
            if ~isnan(rtms)
                rts = rtms / 1000;
            end
        end
        doorTrials(i).reaction_time_s = rts;

        doorTrials(i).risk_key   = str("risk_key", i);
        doorTrials(i).risk_value = num("risk_value", i);

        doorTrials(i).decision_timeout_ms_used = num("decision_timeout_ms_used", i);

        % Optional visual asset information.
        doorTrials(i).background_src = str("background_src", i);
        doorTrials(i).door_src       = str("door_src", i);
        doorTrials(i).victim_skin    = str("victim_skin", i);
    end
end

% -------------------------------------------------------------------------
% Local helper functions (numeric, string, logical getters)
% -------------------------------------------------------------------------

function v = get_num(T, name, idx)
    % GET_NUM  Retrieve numeric value from a table column with coercion.
    if ~ismember(name, T.Properties.VariableNames)
        v = NaN;
        return;
    end
    a = T.(name)(idx);
    if ismissing(a) || (isstring(a) && a == "")
        v = NaN;
        return;
    end
    if isnumeric(a)
        v = double(a);
    elseif islogical(a)
        v = double(a);
    else
        v = str2double(string(a));
    end
end

function v = get_str(T, name, idx)
    % GET_STR  Retrieve string value from a table column with defaults.
    if ~ismember(name, T.Properties.VariableNames)
        v = "";
        return;
    end
    a = T.(name)(idx);
    if ismissing(a)
        v = "";
        return;
    end
    v = string(a);
end

function v = get_log(T, name, idx)
    % GET_LOG  Retrieve logical value from a table column with coercion.
    %
    % Strings are interpreted as true if they match any of:
    %   "true", "1", "yes", "y" (case-insensitive).
    if ~ismember(name, T.Properties.VariableNames)
        v = false;
        return;
    end
    a = T.(name)(idx);
    if islogical(a)
        v = a;
        return;
    end
    if isnumeric(a)
        v = (a ~= 0);
        return;
    end
    s = lower(strtrim(string(a)));
    v = (s == "true" | s == "1" | s == "yes" | s == "y");
end
