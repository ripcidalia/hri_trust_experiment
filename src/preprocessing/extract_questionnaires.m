function questionnaires = extract_questionnaires(Tk)
% extract_questionnaires  Extract precomputed questionnaire scores.
%
%   questionnaires = extract_questionnaires(Tk)
%
% This function collects questionnaire scores from a participant-level
% event table Tk. It relies on precomputed numeric scores stored in the
% 'response' field, with the following conventions:
%
%   - 14-item questionnaire:
%       response = "56.42857143"
%       -> total_percent
%
%   - 40-item questionnaire:
%       response = "(49.8, 50)"
%       -> total_percent                 = 49.8
%          trust14_equiv_total_percent   = 50
%
% For each questionnaire type, the function returns a scalar struct:
%   questionnaires.t40_pre
%   questionnaires.t40_post
%   questionnaires.t14_mid1
%   questionnaires.t14_mid2
%
% Each struct contains parsed totals and a 'raw' field with the raw
% item-level data if available (from qa_pairs_json or response_struct).
%
% Inputs:
%   Tk - Table containing experiment events for one participant. Expected
%        to include 'event_type', 'response', and optionally
%        'qa_pairs_json' and 'response_struct'.
%
% Outputs:
%   questionnaires - Struct with fields:
%       t40_pre, t40_post, t14_mid1, t14_mid2
%     Each field may be [] if the corresponding event is not present.

    questionnaires = struct();

    % Map table event types -> struct field names.
    map = {
        "questionnaire40pre",  "t40_pre";
        "questionnaire40post", "t40_post";
        "questionnaire14mid1", "t14_mid1";
        "questionnaire14mid2", "t14_mid2";
    };

    % For each questionnaire event type, extract a scalar result struct.
    for i = 1:size(map, 1)
        ev = map{i, 1};
        nm = map{i, 2};
        questionnaires.(nm) = local_one(Tk, ev);
    end
end

% -------------------------------------------------------------------------
% Local helpers
% -------------------------------------------------------------------------

function Q = local_one(Tk, eventType)
    % LOCAL_ONE  Extract a single questionnaire instance of a given type.
    %
    % Returns [] if the event does not exist. Otherwise returns a scalar
    % struct with parsed totals and raw item information.

    % Check if the requested event type exists.
    mask = (string(Tk.event_type) == eventType);
    if ~any(mask)
        Q = [];
        return;
    end

    % Use the first occurrence of this event type.
    r = find(mask, 1, 'first');

    % Pull the response string (precomputed score(s)).
    respStr = "";
    if ismember("response", Tk.Properties.VariableNames)
        respStr = string(Tk.response(r));
    end

    % Extract raw item-level data if present (from qa_pairs_json or
    % response_struct).
    raw = [];
    if ismember("qa_pairs_json", Tk.Properties.VariableNames)
        raw = safejsondecode(Tk.qa_pairs_json(r));
    end
    if isempty(raw) && ismember("response_struct", Tk.Properties.VariableNames)
        rs = Tk.response_struct{r};
        if isstruct(rs)
            raw = rs;
        end
    end

    % Prevent unintended struct-array expansion: if raw is non-scalar,
    % store it inside a cell so that Q remains a scalar struct.
    rawStored = raw;
    if ~(isempty(raw) || isscalar(raw))
        rawStored = {raw};
    end

    % Build scalar struct with parsed totals based on questionnaire type.
    if any(eventType == ["questionnaire14mid1","questionnaire14mid2"])
        % 14-item mid-block questionnaire: single score.
        total = parse_single_score(respStr);
        Q = struct( ...
            'total_percent', total, ...
            'raw', rawStored ...
        );

    elseif any(eventType == ["questionnaire40pre","questionnaire40post"])
        % 40-item questionnaire: pair of scores (40-item and 14-item-equivalent).
        [s40, s14eq] = parse_pair_scores(respStr);
        Q = struct( ...
            'total_percent', s40, ...
            'trust14_equiv_total_percent', s14eq, ...
            'raw', rawStored ...
        );

    else
        % Fallback for unexpected event types.
        Q = struct('total_percent', NaN, 'raw', rawStored);
    end
end

function v = parse_single_score(s)
% PARSE_SINGLE_SCORE  Robustly parse a single numeric score from text.
%
%   v = parse_single_score(s)
%
% Handles a variety of input formats, for example:
%   '56.428'
%   '"56.428"'
%   '["56.428"]'
%   '56.428%'
% and returns a numeric scalar (double). Returns NaN if parsing fails.

    v = NaN;

    % If input is cell or numeric, unwrap to a simpler representation.
    if iscell(s)
        if ~isempty(s)
            s = s{1};
        else
            return;
        end
    end
    if isnumeric(s)
        if isscalar(s)
            v = double(s);
        end
        return;
    end

    % Convert to string and strip surrounding quotes/brackets/whitespace.
    s = string(s);
    s = strip(strip(s, '"'), "'");
    s = regexprep(s, '^\s*[\(\[\{"]\s*', '');
    s = regexprep(s, '\s*[\)\]\}"]\s*$', '');

    % If there appears to be a comma-separated representation, attempt to
    % extract the first numeric-looking token.
    if contains(s, ",")
        tokens = regexp(s, '(-?\d+(\.\d+)?([eE][+\-]?\d+)?)', 'match');
        if ~isempty(tokens)
            v = str2double(tokens{1});
            return;
        end
    end

    % Otherwise, strip non-numeric characters and convert.
    s = regexprep(s, '[^0-9\.\-eE]+', '');
    vTry = str2double(s);
    if ~isnan(vTry)
        v = vTry;
    end
end

function [a, b] = parse_pair_scores(s)
% PARSE_PAIR_SCORES  Parse a pair of numeric scores from text.
%
%   [a, b] = parse_pair_scores(s)
%
% Handles a variety of formats, for example:
%   "(49.8, 50)"
%   '"(49.8,50)"'
%   "[49.8,50]"
% and other messy variants. Returns NaN for elements that cannot be parsed.

    a = NaN;
    b = NaN;

    % Handle cell or numeric forms first.
    if iscell(s)
        if ~isempty(s)
            s = s{1};
        else
            return;
        end
    end
    if isnumeric(s)
        % If numeric vector with at least 2 elements, accept first two.
        if numel(s) >= 2
            a = double(s(1));
            b = double(s(2));
        end
        return;
    end

    % Normalize to string and strip surrounding quotes/brackets/whitespace.
    s = string(s);
    s = strip(strip(s, '"'), "'");
    s = regexprep(s, '^\s*[\(\[\{"]\s*', '');
    s = regexprep(s, '\s*[\)\]\}"]\s*$', '');

    % Attempt to split on comma and extract two numeric values.
    parts = split(s, ',');
    if numel(parts) >= 2
        a = str2double(regexprep(strtrim(parts(1)), '[^0-9\.\-eE]+', ''));
        b = str2double(regexprep(strtrim(parts(2)), '[^0-9\.\-eE]+', ''));
    else
        % Fallback: search for any two numeric tokens in the string.
        nums = regexp(s, '(-?\d+(\.\d+)?([eE][+\-]?\d+)?)', 'match');
        if numel(nums) >= 2
            a = str2double(nums{1});
            b = str2double(nums{2});
        elseif numel(nums) == 1
            a = str2double(nums{1});
        end
    end
end
