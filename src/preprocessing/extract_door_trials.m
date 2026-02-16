function doorTrials = extract_door_trials(Tk)
% extract_door_trials  Build struct array of door_trial events from table.
%
%   doorTrials = extract_door_trials(Tk)
%
% This function extracts all rows corresponding to door trials
% (event_type == "door_trial") from the participant-level table Tk and
% converts them into a 1×N struct array. Each struct aggregates relevant
% information about a single door trial, including:
%   - Block and trial indices
%   - Suggestion vs. participant choice
%   - Followed/correct/timed-out flags
%   - Reaction time
%   - Risk information
%   - Visual asset metadata (background_src, door_src, victim_skin)
%
% Inputs:
%   Tk         - Table containing all events for a single participant.
%                It must include at least the variable event_type, and
%                typically the additional variables accessed below
%                (block_index, trial_index, suggestion, choice, etc.).
%
% Outputs:
%   doorTrials - 1×N struct array, where N is the number of door_trial
%                events found. If no such events exist, doorTrials is an
%                empty struct array.
%
% Notes:
%   - Various helper getters (get_num, get_str, get_log) are used to
%     robustly handle missing fields, missing values, and type coercion
%     (string/char/logical/numeric).
%   - Reaction time is taken from reaction_time_s when available; if
%     missing, it falls back to rt_ms / 1000.

    % Select only door_trial rows.
    mask = (string(Tk.event_type) == "door_trial");
    D = Tk(mask, :);

    % If there are no door trials, return an empty struct array.
    if isempty(D)
        doorTrials = struct([]);
        return;
    end

    % Preallocate struct array with default fields and types.
    N = height(D);
    doorTrials = repmat(struct( ...
        'block_index', NaN, ...
        'trial_index', NaN, ...
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
        'victim_skin', "" ), N, 1);

    % Helper function handles (numeric, string, logical) for a given column
    % and row index.
    num  = @(col, idx) get_num(D, col, idx);
    str  = @(col, idx) get_str(D, col, idx);
    logi = @(col, idx) get_log(D, col, idx);

    % Populate each doorTrials entry from the corresponding row in D.
    for i = 1:N
        doorTrials(i).block_index = num("block_index", i);
        doorTrials(i).trial_index = num("trial_index", i);
        doorTrials(i).suggestion  = str("suggestion", i);
        doorTrials(i).choice      = str("choice", i);
        doorTrials(i).followed    = logi("followed", i);
        doorTrials(i).correct     = logi("correct", i);
        doorTrials(i).timed_out   = logi("timed_out", i);

        % Reaction time preference: reaction_time_s -> rt_ms / 1000 (if needed).
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
% Local helper functions for robust field access
% -------------------------------------------------------------------------

function v = get_num(T, name, idx)
    % GET_NUM  Retrieve numeric value from T{name}(idx) with coercion.
    %
    % If the column does not exist or the value is missing/empty, NaN is
    % returned. Strings and chars are converted using str2double, logicals
    % are cast to double.
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
    % GET_STR  Retrieve string value from T{name}(idx) with defaults.
    %
    % If the column does not exist or the value is missing, returns an
    % empty string ("").
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
    % GET_LOG  Retrieve logical value from T{name}(idx) with coercion.
    %
    % If the column does not exist, returns false. Numeric values are
    % interpreted as (a ~= 0). Strings are interpreted using a small set
    % of truthy tokens: "true", "1", "yes", "y" (case-insensitive).
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
