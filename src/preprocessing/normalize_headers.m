function T = normalize_headers(T)
% normalize_headers  Standardize table variable names for preprocessing.
%
%   T = normalize_headers(T)
%
% This function standardizes the variable names of a table by applying a
% consistent normalization scheme. It is intended for use during the early
% preprocessing stages of the event data pipeline, where variable names may
% include inconsistent capitalization, spacing, or special characters.
%
% The normalization rules applied:
%   1) Convert all characters to lowercase.
%   2) Trim leading and trailing whitespace.
%   3) Replace internal whitespace with underscores.
%   4) Remove all characters except lowercase letters, digits, and
%      underscores.
%   5) If a resulting name is empty, assign a fallback name "var<i>".
%
% Inputs:
%   T - Table whose variable names are to be normalized.
%
% Outputs:
%   T - The same table, with updated and normalized variable names.

    V = T.Properties.VariableNames;

    for i = 1:numel(V)
        v = V{i};

        % Convert to lowercase and trim whitespace.
        v = lower(v);
        v = strtrim(v);

        % Replace one or more whitespace characters with a single underscore.
        v = regexprep(v, '\s+', '_');

        % Strip any remaining characters outside the allowed set.
        v = regexprep(v, '[^a-z0-9_]', '');

        % Fallback name if the result is empty.
        if isempty(v)
            v = "var" + i;
        end

        V{i} = v;
    end

    T.Properties.VariableNames = V;
end
