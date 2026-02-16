function val = get_string_field(T, name)
% get_string_field  Return the first non-missing, non-empty string value in column 'name'.
%
%   val = get_string_field(T, name)
%
% Scans the column T.(name) from top to bottom and returns the first entry
% that can be interpreted as a *meaningful* string:
%
%   • Not missing (not <missing>)
%   • Not empty ("")
%
% All values are coerced via string(), so this handles chars, numbers,
% categoricals, etc. gracefully.
%
% If no such entry exists, returns "".
%
% Inputs:
%   T    : A table.
%   name : Column name to inspect (string or char).
%
% Output:
%   val  : First non-empty string or "" if none found.

    val = "";

    % Column must exist
    if ~ismember(name, T.Properties.VariableNames)
        return;
    end

    col = T.(name);

    % Linear scan for first meaningful string
    for i = 1:numel(col)
        s = string(col(i));

        if ~(ismissing(s) || s == "")
            val = s;
            return;
        end
    end
end
