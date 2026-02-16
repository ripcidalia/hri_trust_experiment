function T = project_common_fields(T)
% project_common_fields  Flatten commonly used fields into table columns.
%
%   T = project_common_fields(T)
%
% This function projects useful fields from decoded JSON payloads
% (stored in 'extra_struct' and 'response_struct') into flat table
% columns, and applies basic type coercions. It also derives a unified
% reaction_time_s from rt_ms when needed.
%
% Typical usage within Step 1:
%   - Ensure that key variables (event_type, block_index, trial_index,
%     risk_value, review_condition, etc.) exist in the table.
%   - Coerce numeric columns stored as strings into numeric arrays.
%   - Extract door- and review-related fields from nested structs.
%   - Normalize booleans (followed, correct, timed_out) to logical.
%
% Inputs:
%   T - Event-level table for all participants, including (possibly)
%       'extra_struct' and 'response_struct' columns containing decoded
%       JSON for each row.
%
% Outputs:
%   T - Same table, with guaranteed presence and cleaned types for
%       commonly used columns, suitable for downstream processing.

    % ---------------------------------------------------------------------
    % 1) Ensure common columns exist (with appropriate default types)
    % ---------------------------------------------------------------------
    T = ensure_vars_exist(T, {
        'event_type',               strings(0,1)
        'block_index',              NaN
        'trial_index',              NaN
        'suggestion',               strings(0,1)
        'choice',                   strings(0,1)
        'followed',                 false
        'correct',                  false
        'timed_out',                false
        'reaction_time_s',          NaN
        'decision_timeout_ms_used', NaN
        'risk_key',                 strings(0,1)
        'risk_value',               NaN
        'review_condition',         strings(0,1)
        'review_expected',          NaN
        'review_ids',               strings(0,1)
        'review_tones',             strings(0,1)
        'review_avatars',           strings(0,1)
        'background_src',           strings(0,1)
        'door_src',                 strings(0,1)
        'victim_skin',              strings(0,1)
        'ts_seq',                   NaN
        'rt_ms',                    NaN
        'device_type',              strings(0,1)
        'browser_name',             strings(0,1)
        'browser_major',            NaN
        'client_version',           strings(0,1)
        'participant_id',           strings(0,1)
        'session_id',               strings(0,1)
        'set_id',                   strings(0,1)
        'seed',                     NaN
    });

    % ---------------------------------------------------------------------
    % 2) Coerce selected columns from string/cell to numeric where possible
    % ---------------------------------------------------------------------
    T.ts_seq         = local_tonumber(T.ts_seq);
    T.block_index    = local_tonumber(T.block_index);
    T.trial_index    = local_tonumber(T.trial_index);
    T.browser_major  = local_tonumber(T.browser_major);
    T.seed           = local_tonumber(T.seed);
    T.rt_ms          = local_tonumber(T.rt_ms);
    T.risk_value     = local_tonumber(T.risk_value);
    T.review_expected= local_tonumber(T.review_expected);

    % ---------------------------------------------------------------------
    % 3) Derive reaction_time_s from rt_ms where reaction_time_s is missing
    % ---------------------------------------------------------------------
    needRT = isnan(T.reaction_time_s) & ~isnan(T.rt_ms);
    T.reaction_time_s(needRT) = T.rt_ms(needRT) ./ 1000;

    % ---------------------------------------------------------------------
    % 4) Row-wise projection from extra_struct and response_struct
    % ---------------------------------------------------------------------
    % For each row, copy relevant fields from the decoded JSON payloads into
    % the flat columns created above, if those fields are present.
    for i = 1:height(T)
        xs = T.extra_struct{i};
        rs = T.response_struct{i};

        % ----- Extract from extra_struct (if it is a struct) -----
        if isstruct(xs)
            T = assign_if_present(T, i, xs, 'suggestion');
            T = assign_if_present(T, i, xs, 'choice');
            T = assign_if_present(T, i, xs, 'followed');
            T = assign_if_present(T, i, xs, 'correct');
            T = assign_if_present(T, i, xs, 'timed_out');
            T = assign_if_present(T, i, xs, 'reaction_time_s');
            T = assign_if_present(T, i, xs, 'decision_timeout_ms_used');
            T = assign_if_present(T, i, xs, 'risk_key');
            T = assign_if_present(T, i, xs, 'risk_value');
            T = assign_if_present(T, i, xs, 'background_src');
            T = assign_if_present(T, i, xs, 'door_src');
            T = assign_if_present(T, i, xs, 'victim_skin');
            T = assign_if_present(T, i, xs, 'review_condition');
            T = assign_if_present(T, i, xs, 'review_expected');
            T = assign_if_present(T, i, xs, 'review_ids');
            T = assign_if_present(T, i, xs, 'review_tones');
            T = assign_if_present(T, i, xs, 'review_avatars');
            T = assign_if_present(T, i, xs, 'seed');
        end

        % ----- Extract from response_struct (if it is a struct) -----
        if isstruct(rs)
            % Seed (if present in the same struct as used in upstream code)
            T = assign_if_present(T, i, xs, 'seed');

            % Door-trial related payload fields
            T = assign_if_present(T, i, rs, 'suggestion');
            T = assign_if_present(T, i, rs, 'choice');
            T = assign_if_present(T, i, rs, 'followed');
            T = assign_if_present(T, i, rs, 'correct');
            T = assign_if_present(T, i, rs, 'timed_out');
            T = assign_if_present(T, i, rs, 'reaction_time_s');
            T = assign_if_present(T, i, rs, 'decision_timeout_ms_used');

            % Review-related fields
            T = assign_if_present(T, i, rs, 'review_condition');
            T = assign_if_present(T, i, rs, 'review_expected');
            T = assign_if_present(T, i, rs, 'review_ids');
            T = assign_if_present(T, i, rs, 'review_tones');
            T = assign_if_present(T, i, rs, 'review_avatars');

            % Risk information (if present in response_struct)
            T = assign_if_present(T, i, rs, 'risk_key');
            T = assign_if_present(T, i, rs, 'risk_value');

            % Visual asset references
            T = assign_if_present(T, i, rs, 'background_src');
            T = assign_if_present(T, i, rs, 'door_src');
            T = assign_if_present(T, i, rs, 'victim_skin');
        end
    end

    % ---------------------------------------------------------------------
    % 5) Logical coercions for followed / correct / timed_out
    % ---------------------------------------------------------------------
    % Convert string/numeric encodings ("true", "yes", 1, 0, etc.) to
    % logical true/false.
    T.followed  = local_tologic(T.followed);
    T.correct   = local_tologic(T.correct);
    T.timed_out = local_tologic(T.timed_out);

    % ---------------------------------------------------------------------
    % 6) Final numeric coercions for reaction_time_s and timeouts
    % ---------------------------------------------------------------------
    T.decision_timeout_ms_used = local_tonumber(T.decision_timeout_ms_used);
    T.reaction_time_s          = local_tonumber(T.reaction_time_s);
end

% =========================================================================
% Local helper functions
% =========================================================================

function T = assign_if_present(T, i, S, field)
% ASSIGN_IF_PRESENT  Copy S.(field) into table T.(field)(i) if available.
%
% This helper checks whether a given field exists in struct S and is
% non-empty. If so, its value is written into the corresponding table
% column at row i, with simple normalization for char to string.
    if isfield(S, field) && ~isempty(S.(field))
        val = S.(field);
        % Normalize character arrays to string.
        if ischar(val)
            val = string(val);
        end
        % Direct assignment into the table column.
        T.(field)(i) = val;
    end
end

function x = local_tonumber(x)
% LOCAL_TONUMBER  Coerce various types into a numeric array where possible.
%
% Handles:
%   - string / cellstr: uses str2double.
%   - cell: attempts to parse each entry individually.
%   - logical: cast to double.
%   - other non-numeric: best-effort cast to double; if this fails,
%     a NaN array is created.

    if isstring(x) || iscellstr(x)
        x = str2double(string(x));

    elseif iscell(x)
        % Try to coerce cell arrays of mixed types.
        tmp = nan(size(x));
        for k = 1:numel(x)
            v = x{k};
            if isstring(v) || ischar(v)
                tmp(k) = str2double(string(v));
            elseif isnumeric(v)
                tmp(k) = v;
            else
                tmp(k) = NaN;
            end
        end
        x = tmp;

    elseif islogical(x)
        x = double(x);

    elseif ~isnumeric(x)
        % Fallback: attempt to cast to double.
        try
            x = double(x);
        catch
            x = nan(height(x), 1); %#ok<NASGU>
        end
    end
end

function y = local_tologic(y)
% LOCAL_TOLOGIC  Coerce strings/numbers/cells into logical flags.
%
% True values for string input are any of:
%   "true", "1", "yes", "y" (case-insensitive).
% Numeric input is true if non-zero.
% Cell arrays are processed element-wise.

    if isstring(y)
        s = lower(strtrim(y));
        y = (s == "true") | (s == "1") | (s == "yes") | (s == "y");

    elseif isnumeric(y)
        y = (y ~= 0);

    elseif iscell(y)
        tmp = false(size(y));
        for k = 1:numel(y)
            v = y{k};
            if isstring(v) || ischar(v)
                s = lower(strtrim(string(v)));
                tmp(k) = (s == "true") || (s == "1") || (s == "yes") || (s == "y");
            elseif isnumeric(v)
                tmp(k) = (v ~= 0);
            elseif islogical(v)
                tmp(k) = v;
            end
        end
        y = tmp;

    elseif islogical(y)
        % Already logical; leave unchanged.

    else
        % Unknown type: default to false.
        y = false(size(y));
    end
end
