function demographics = extract_demographics(Tk)
% extract_demographics  Decode demographics information from event table.
%
%   demographics = extract_demographics(Tk)
%
% This function extracts demographics information for a single participant
% from their event table Tk. It expects a single "demographics" event whose
% response (typically JSON-encoded) contains fields such as age range and
% gender. The decoded JSON is stored in full under demographics.raw, and
% commonly used fields are exposed as top-level strings.
%
% Extracted fields:
%   raw       : full decoded demographics structure (or []).
%   age_range : string label of age range (e.g., "18-24", "25-34").
%   gender    : string label of gender after applying self-describe logic.
%
% Gender handling:
%   - If a gender option field (gender_option / genderOption) is
%     "self_describe" or "self-describe", gender_self_desc (or similar)
%     is used as the final gender string.
%   - Otherwise, the gender_option field itself is used if non-empty.
%   - If no option is found, the function falls back to any generic
%     "gender" field present in the raw JSON.
%
% Inputs:
%   Tk - Table containing all events for a single participant. Must include
%        'event_type' and either a decoded 'response_struct' column or a
%        'response' column that can be decoded as JSON.
%
% Outputs:
%   demographics - Struct with fields:
%                    raw       : decoded demographics struct or []
%                    age_range : string (possibly empty)
%                    gender    : string (possibly empty)

    % ---------------------------------------------------------------------
    % 1) Locate demographics event row
    % ---------------------------------------------------------------------
    mask = (string(Tk.event_type) == "demographics");
    if ~any(mask)
        % If no demographics event is present, return minimal struct.
        demographics = struct('raw', [], ...
                              'age_range', "", ...
                              'gender', "");
        return;
    end

    % Use the first demographics row if multiple are present.
    r = find(mask, 1, 'first');

    % ---------------------------------------------------------------------
    % 2) Decode raw demographics JSON (if available)
    % ---------------------------------------------------------------------
    raw = [];
    if ismember("response_struct", Tk.Properties.VariableNames)
        rs = Tk.response_struct{r};
        if isstruct(rs)
            raw = rs;
        end
    end
    if isempty(raw) && ismember("response", Tk.Properties.VariableNames)
        % Fallback: decode JSON from response text if response_struct is not set.
        raw = safejsondecode(Tk.response(r));
    end

    % Initialize output struct with defaults.
    demographics = struct('raw', raw, ...
                          'age_range', "", ...
                          'gender', "");

    % ---------------------------------------------------------------------
    % 3) Extract age_range (simple string field)
    % ---------------------------------------------------------------------
    demographics.age_range = field_str(raw, ["age_range","AgeRange","agerange"]);

    % ---------------------------------------------------------------------
    % 4) Extract gender with self-describe handling
    % ---------------------------------------------------------------------
    % Gender option field (e.g., "woman", "man", "self_describe", etc.).
    gopt = field_str(raw, ["gender_option","genderOption"]);

    if gopt == "self_describe" || gopt == "self-describe"
        % Self-describe path: use free-text field for gender.
        gself = field_str(raw, ["gender_self_desc","genderSelfDesc","gender_free"]);
        demographics.gender = gself;
    elseif gopt ~= ""
        % Use the option label directly if present.
        demographics.gender = gopt;
    else
        % Fallback: look for any generic 'gender' field.
        demographics.gender = field_str(raw, ["gender","Gender"]);
    end
end

% -------------------------------------------------------------------------
% Local helper: extract string-like field from a struct
% -------------------------------------------------------------------------

function v = field_str(S, names)
    % FIELD_STR  Return first string-like field value from struct S.
    %
    % Iterates over 'names' and returns the first field found as a string.
    % Numeric values are converted to string. Returns "" if no fields are
    % present or if S is empty.
    v = "";
    if isempty(S)
        return;
    end
    for n = names
        if isfield(S, n)
            x = S.(n);
            if isstring(x) || ischar(x)
                v = string(x);
                return;
            end
            if isnumeric(x)
                v = string(x);
                return;
            end
        end
    end
end
