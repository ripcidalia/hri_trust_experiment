function val = get_numeric_field(T, name)
% get_numeric_field  Return the first non-missing numeric value in column 'name'.
%
%   val = get_numeric_field(T, name)
%
% Searches the column T.(name) top-to-bottom and returns the first entry
% that can be interpreted as a valid numeric scalar. Accepted formats:
%   - Numerics (double, single, integer types)
%   - Logical (converted to 0/1)
%   - Strings/chars convertible via str2double
%
% If no valid numeric value is found, returns NaN.
%
% Inputs:
%   T    : A table.
%   name : Column name to inspect (string or char).
%
% Output:
%   val  : First valid numeric from the column, or NaN if none found.

    val = NaN;

    % Column must exist
    if ~ismember(name, T.Properties.VariableNames)
        return;
    end

    col = T.(name);

    % Linear scan: pick first interpretable numeric value
    for i = 1:numel(col)
        x = col(i);

        % Case 1: already numeric or logical
        if isnumeric(x) || islogical(x)
            if ~isnan(x)
                val = double(x);
                return;
            end
            continue;  % numeric but NaN → skip
        end

        % Case 2: string/char → convert
        xs = str2double(string(x));
        if ~isnan(xs)
            val = xs;
            return;
        end
    end
end
