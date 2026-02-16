function trustProbes = extract_trust_probes(Tk)
% extract_trust_probes  Extract mid-block trust probes from event table.
%
%   trustProbes = extract_trust_probes(Tk)
%
% This function extracts all events of type "trust_probe_mid" from a
% participant-level event table and converts them into a 1×M struct array.
% Each trust probe is associated with:
%   - block_index : block index from the raw data (typically 0-based).
%   - trial_index : global trial index from the raw data (typically 0-based).
%   - value       : trust rating on a 0–100 scale, if it can be parsed.
%
% The trust value is inferred using the following priority:
%   1) response_struct.value / .trust / .answer / .slider / .rating
%   2) response field interpreted as a numeric string
%   3) extra_struct.value / .trust / .answer / .slider / .rating
%
% If no numeric value can be extracted, the probe value is set to NaN.
%
% Inputs:
%   Tk          - Table of events for a single participant. Must contain:
%                   event_type, block_index, trial_index,
%                 and may contain:
%                   response_struct, response, extra_struct.
%
% Outputs:
%   trustProbes - 1×M struct array of trust probe entries. If no
%                 "trust_probe_mid" events exist, an empty struct array
%                 is returned.

    % Select only "trust_probe_mid" rows.
    mask = (string(Tk.event_type) == "trust_probe_mid");
    U = Tk(mask, :);

    % No mid-block trust probes for this participant.
    if isempty(U)
        trustProbes = struct([]);
        return;
    end

    % Preallocate struct array with basic fields.
    M = height(U);
    trustProbes = repmat(struct( ...
        'block_index', NaN, ...
        'trial_index', NaN, ...
        'value', NaN ), M, 1);

    % Populate each trust probe.
    for i = 1:M
        % Block and trial indices from the raw columns (typically 0-based).
        trustProbes(i).block_index = firstnum(U, i, "block_index");
        trustProbes(i).trial_index = firstnum(U, i, "trial_index");

        % Try to find a 0–100 trust value.
        % Sources, in order of preference:
        %   1) response_struct (decoded JSON)
        %   2) response (plain text that might be numeric)
        %   3) extra_struct (decoded JSON)
        v = [];

        % 1) response_struct: struct decoded from JSON in response
        if ismember("response_struct", U.Properties.VariableNames)
            RS = U.response_struct{i};
            if isstruct(RS)
                cand = pick_first_numeric(RS, ["value","trust","answer","slider","rating"]);
                if ~isempty(cand)
                    v = cand;
                end
            end
        end

        % 2) response: sometimes directly stores a numeric value as text
        if isempty(v) && ismember("response", U.Properties.VariableNames)
            vtry = str2double(string(U.response(i)));
            if ~isnan(vtry)
                v = vtry;
            end
        end

        % 3) extra_struct: alternative JSON location for the numeric value
        if isempty(v) && ismember("extra_struct", U.Properties.VariableNames)
            XS = U.extra_struct{i};
            if isstruct(XS)
                cand = pick_first_numeric(XS, ["value","trust","answer","slider","rating"]);
                if ~isempty(cand)
                    v = cand;
                end
            end
        end

        if isempty(v)
            v = NaN;
        end
        trustProbes(i).value = v;
    end
end

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------

function v = firstnum(T, i, name)
    % FIRSTNUM  Extract a numeric value from T.(name)(i) with coercion.
    %
    % If the column does not exist, returns NaN. If the entry is numeric,
    % it is cast to double; otherwise, an attempt is made to parse it as a
    % numeric string.
    if ~ismember(name, T.Properties.VariableNames)
        v = NaN;
        return;
    end
    x = T.(name)(i);
    if isnumeric(x)
        v = double(x);
    else
        v = str2double(string(x));
    end
end

function n = pick_first_numeric(S, names)
    % PICK_FIRST_NUMERIC  Return the first numeric-looking field in S.
    %
    % Given a struct S and an ordered list of field names, this function
    % searches for the first field present in S that can be interpreted as
    % numeric. Supported cases:
    %   - Numeric value (returned as double).
    %   - String/char that can be parsed via str2double.
    %
    % If no numeric field is found, returns [].
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
