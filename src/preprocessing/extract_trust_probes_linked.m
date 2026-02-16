function trustProbes = extract_trust_probes_linked(Tk, door_order_index)
% extract_trust_probes_linked  Extract trust probes and link them to context.
%
%   trustProbes = extract_trust_probes_linked(Tk, door_order_index)
%
% This function builds a struct array of mid-block trust probes
% ("trust_probe_mid") and links each probe either:
%   - to the questionnaire immediately preceding it, or
%   - to the nearest preceding door trial.
%
% Linking rules:
%   1) after_questionnaire:
%        probe.trial_index_raw == questionnaire.trial_index_raw + 1
%   2) after_door:
%        otherwise, link to the door_trial with the largest trial_index_raw
%        that is still strictly smaller than the probe's trial_index_raw.
%
% For each probe, the following fields are populated:
%
% Common:
%   origin                   : "after_questionnaire" | "after_door"
%   trial_index_raw          : probe raw global trial_index (0-based)
%   value                    : numeric 0–100 if available, otherwise NaN
%
% When origin = "after_questionnaire":
%   questionnaire_event_type : original questionnaire event_type
%                              (e.g., "questionnaire14mid1")
%   questionnaire_type       : normalized type label:
%                                 "t14_mid1", "t14_mid2",
%                                 "t40_pre",  "t40_post"
%   questionnaire_raw_index  : raw questionnaire trial_index (0-based)
%   block_index              : NaN
%   trial_index              : NaN
%   linked_door_trial_index_raw : NaN
%   linked_door_id           : ""
%
% When origin = "after_door":
%   block_index              : 1-based block index of linked door
%   trial_index              : within-block door order (1..N) of linked door
%   linked_door_trial_index_raw : linked door's raw global trial_index (0-based)
%   linked_door_id           : "door_B_K" (B = block, K = within-block order)
%   questionnaire_* fields   : empty / NaN
%
% Inputs:
%   Tk               - Table of events for a single participant, including
%                      event_type, block_index, trial_index, and optional
%                      response/extra_struct columns used to extract probe values.
%
%   door_order_index - Column vector (same height as Tk) giving, for door
%                      trials, the within-block door order index (1..N). For
%                      non-door rows, values are typically NaN.
%
% Outputs:
%   trustProbes      - 1×M struct array of linked trust probe entries. If
%                      no "trust_probe_mid" events are present, an empty
%                      struct array is returned.

    % ---------------------------------------------------------------------
    % 1) Core columns and event type classification
    % ---------------------------------------------------------------------
    et   = string(Tk.event_type);
    bi0  = get_num_col(Tk, "block_index");   % 0-based block index in logs
    ti0  = get_num_col(Tk, "trial_index");   % 0-based global trial index in logs

    % Identify event categories
    isDoor  = (et == "door_trial");
    isProbe = (et == "trust_probe_mid");
    isQ     = ismember(et, [ ...
        "questionnaire14mid1", ...
        "questionnaire14mid2", ...
        "questionnaire40pre", ...
        "questionnaire40post"]);

    % ---------------------------------------------------------------------
    % 2) Extract door trial information needed for linking
    % ---------------------------------------------------------------------
    door_rows = find(isDoor);
    door_b1   = bi0(door_rows) + 1;          % 1-based block index
    door_ti0  = ti0(door_rows);              % raw global trial index (0-based)
    door_k    = door_order_index(door_rows); % within-block door order (1..N)

    % Build door identifier strings "door_B_K"
    door_id = strings(numel(door_rows), 1);
    for d = 1:numel(door_rows)
        if ~isnan(door_b1(d)) && ~isnan(door_k(d))
            door_id(d) = "door_" + string(door_b1(d)) + "_" + string(door_k(d));
        else
            door_id(d) = "";
        end
    end

    % ---------------------------------------------------------------------
    % 3) Extract questionnaire information for questionnaire-linked probes
    % ---------------------------------------------------------------------
    q_rows = find(isQ);
    q_ti0  = ti0(q_rows);        % raw trial index (0-based)
    q_et   = et(q_rows);         % original event_type

    % Map event_type -> normalized questionnaire_type
    q_type = strings(numel(q_rows), 1);
    for i = 1:numel(q_rows)
        switch q_et(i)
            case "questionnaire14mid1"
                q_type(i) = "t14_mid1";
            case "questionnaire14mid2"
                q_type(i) = "t14_mid2";
            case "questionnaire40pre"
                q_type(i) = "t40_pre";
            case "questionnaire40post"
                q_type(i) = "t40_post";
            otherwise
                q_type(i) = "";
        end
    end

    % ---------------------------------------------------------------------
    % 4) Identify probes and preallocate output struct
    % ---------------------------------------------------------------------
    Urows = find(isProbe);
    if isempty(Urows)
        trustProbes = struct([]);
        return;
    end

    M = numel(Urows);
    trustProbes = repmat(struct( ...
        'origin', "", ...
        'trial_index_raw', NaN, ...
        'questionnaire_event_type', "", ...
        'questionnaire_type', "", ...
        'questionnaire_raw_index', NaN, ...
        'block_index', NaN, ...
        'trial_index', NaN, ...
        'linked_door_trial_index_raw', NaN, ...
        'linked_door_id', "", ...
        'value', NaN ), M, 1);

    % ---------------------------------------------------------------------
    % 5) Link each probe to questionnaire or preceding door
    % ---------------------------------------------------------------------
    for i = 1:M
        r  = Urows(i);
        pr = ti0(r);  % probe raw global index (0-based)

        trustProbes(i).trial_index_raw = pr;

        % ----- Case 1: immediately follows a questionnaire (q_raw == pr - 1)
        qIdx = find(q_ti0 == pr - 1, 1, 'last');
        if ~isempty(qIdx)
            trustProbes(i).origin                   = "after_questionnaire";
            trustProbes(i).questionnaire_event_type = q_et(qIdx);
            trustProbes(i).questionnaire_type       = q_type(qIdx);
            trustProbes(i).questionnaire_raw_index  = q_ti0(qIdx);
            % Leave block_index / trial_index / door linkage fields as NaN/""
        else
            % ----- Case 2: link to nearest preceding door (max door_ti0 < pr)
            trustProbes(i).origin = "after_door";

            idx = find(door_ti0 < pr);
            if ~isempty(idx)
                % Select door with maximal trial_index_raw among those < pr.
                [~, rel] = max(door_ti0(idx));
                dIdx = idx(rel);

                trustProbes(i).block_index                 = door_b1(dIdx);
                trustProbes(i).trial_index                 = door_k(dIdx);
                trustProbes(i).linked_door_trial_index_raw = door_ti0(dIdx);
                trustProbes(i).linked_door_id              = door_id(dIdx);
            end
        end

        % Extract numeric probe value (0–100 scale if available).
        trustProbes(i).value = local_extract_value(Tk, r);
    end
end

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------

function x = get_num_col(T, name)
    % GET_NUM_COL  Retrieve numeric column, coercing to double where needed.
    %
    % If the named column exists, attempts to convert it to a numeric
    % vector. If it does not exist, returns a NaN column of height(T).
    if ismember(name, T.Properties.VariableNames)
        x = T.(name);
        if iscell(x),   x = cellfun(@local_num, x); end
        if isstring(x), x = str2double(x);          end
        if ~isnumeric(x), x = double(x);            end
    else
        x = nan(height(T), 1);
    end
end

function v = local_num(a)
    % LOCAL_NUM  Convert mixed-type scalar to numeric value.
    if isnumeric(a)
        v = a;
    elseif isstring(a) || ischar(a)
        v = str2double(string(a));
    elseif islogical(a)
        v = double(a);
    else
        v = NaN;
    end
end

function v = local_extract_value(Tk, row)
    % LOCAL_EXTRACT_VALUE  Extract trust probe value for a single row.
    %
    % Tries, in order:
    %   1) response_struct: struct with fields value/trust/answer/slider/rating
    %   2) response: plain numeric-as-text
    %   3) extra_struct: struct with the same candidate fields
    %
    % Returns NaN if no numeric value can be found.
    v = [];

    % 1) response_struct (decoded JSON)
    if ismember("response_struct", Tk.Properties.VariableNames)
        RS = Tk.response_struct{row};
        if isstruct(RS)
            v = pick_first_numeric(RS, ["value","trust","answer","slider","rating"]);
        end
    end

    % 2) response (plain numeric string)
    if isempty(v) && ismember("response", Tk.Properties.VariableNames)
        vtry = str2double(string(Tk.response(row)));
        if ~isnan(vtry)
            v = vtry;
        end
    end

    % 3) extra_struct (alternative JSON container)
    if isempty(v) && ismember("extra_struct", Tk.Properties.VariableNames)
        XS = Tk.extra_struct{row};
        if isstruct(XS)
            v = pick_first_numeric(XS, ["value","trust","answer","slider","rating"]);
        end
    end

    if isempty(v)
        v = NaN;
    end
end

function n = pick_first_numeric(S, names)
    % PICK_FIRST_NUMERIC  Return the first numeric-like field in struct S.
    %
    % Iterates over 'names' and returns the first field that is numeric or
    % can be parsed as numeric from a string/char. Returns [] if none are
    % suitable.
    n = [];
    for k = 1:numel(names)
        f = names(k);
        if isfield(S, f)
            x = S.(f);
            if isnumeric(x)
                n = double(x);
                return;
            end
            if isstring(x) || ischar(x)
                xv = str2double(string(x));
                if ~isnan(xv)
                    n = xv;
                    return;
                end
            end
        end
    end
end
