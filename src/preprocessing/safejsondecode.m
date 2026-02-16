function out = safejsondecode(txt)
% safejsondecode  Safely decode a JSON string, returning [] on failure.
%
%   out = safejsondecode(txt)
%
% This utility function attempts to decode a JSON-encoded string while
% avoiding errors during preprocessing. It is designed for robustness
% against missing values, malformed JSON, or non-JSON content.
%
% Behaviour:
%   - Accepts string or character input. Converts to string internally.
%   - Trims whitespace and treats empty or missing values as non-JSON.
%   - Only attempts jsondecode if the text begins with '{' or '['.
%   - Catches any jsondecode error and returns [] instead.
%
% Inputs:
%   txt - A string or character array that may contain JSON text.
%
% Outputs:
%   out - Decoded MATLAB value (struct, cell, etc.) if decoding succeeds;
%         otherwise [].

    out = [];

    % Treat empty or missing entries as non-JSON.
    if isempty(txt) || all(ismissing(string(txt)))
        return;
    end

    % Normalize to trimmed string.
    s = string(txt);
    s = strtrim(s);

    % Again handle empty strings or explicit "NaN".
    if s == "" || s == "NaN"
        return;
    end

    % Only attempt JSON decoding if the content appears to be JSON.
    if startsWith(s, ["{", "["])
        try
            out = jsondecode(s);
        catch
            % On any decoding error, return [].
            out = [];
        end
    end
end
