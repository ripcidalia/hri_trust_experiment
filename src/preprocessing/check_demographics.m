function [ok, msg] = check_demographics(P)
% check_demographics  Validate basic structure of demographics information.
%
%   [ok, msg] = check_demographics(P)
%
% This function performs a light-weight consistency check on the
% demographics information stored in P.demographics. The intent is to
% verify that:
%   - If demographics are present, the decoded raw payload is a struct.
%   - age_range and gender, when present, are string-like values.
%
% Importantly, the presence of demographics is not mandatory for inclusion
% in the modelling pipeline; missing demographics are treated as non-fatal.
%
% Inputs:
%   P   - Participant struct that may contain:
%           P.demographics.raw       : decoded JSON struct (or [])
%           P.demographics.age_range : string/char (may be empty)
%           P.demographics.gender    : string/char (may be empty)
%
% Outputs:
%   ok  - Logical flag; false only if demographics are present but not in
%         the expected format (e.g., raw not a struct, or fields not
%         string-like).
%   msg - Diagnostic message (non-empty only when ok == false).

    ok = true;
    msg = "";

    % If no demographics field or it is empty, this is not considered a
    % failure. The participant simply has no demographics event.
    if ~isfield(P, "demographics") || isempty(P.demographics)
        return;
    end

    D = P.demographics;

    % raw must be a struct when present; otherwise something went wrong
    % during JSON decoding.
    if isfield(D, "raw") && ~isempty(D.raw) && ~isstruct(D.raw)
        ok  = false;
        msg = "Demographics raw exists but is not a struct.";
        return;
    end

    % age_range and gender should be string-like (char or string). Empty
    % values are allowed to represent a skipped or "prefer not to say"
    % response.
    if isfield(D, "age_range") && ~(isstring(D.age_range) || ischar(D.age_range))
        ok  = false;
        msg = "Demographics age_range is not a string.";
        return;
    end

    if isfield(D, "gender") && ~(isstring(D.gender) || ischar(D.gender))
        ok  = false;
        msg = "Demographics gender is not a string.";
        return;
    end
end
