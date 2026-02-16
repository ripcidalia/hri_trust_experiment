function T = read_events_table(csvPath)
% read_events_table  Load raw event CSV into a table with robust defaults.
%
%   T = read_events_table(csvPath)
%
% This function reads the raw event log from the experiment and returns a
% MATLAB table with text-like fields converted to strings. It uses
% detectImportOptions to infer variable types and applies optional import
% settings (if supported by the MATLAB version) to preserve whitespace and
% handle empty fields more robustly.
%
% Inputs:
%   csvPath  - Path to the raw CSV file containing event-level data.
%
% Outputs:
%   T        - Table containing the imported event data, with all
%              text-like variables converted to string arrays.
%
% Notes:
%   - The function is designed to be MATLAB-version-agnostic. On older
%     versions, certain import options (e.g., WhitespaceRule, EmptyFieldRule)
%     may not be supported. In those cases, the function silently falls
%     back to default behaviour.
%   - Only type handling and import settings are performed here; no
%     additional data cleaning or preprocessing is applied.

    % Detect basic import settings from the CSV file.
    opts = detectImportOptions(csvPath, "NumHeaderLines", 0);

    % Attempt to configure string-related import options. This may not be
    % supported in all MATLAB releases, so errors are caught and ignored.
    try
        opts = setvaropts(opts, opts.VariableNames, ...
            "WhitespaceRule", "preserve", ...
            "EmptyFieldRule", "auto");
    catch
        % Older MATLAB versions may not support these fields; use defaults.
    end

    % Read the table using the (possibly modified) import options.
    T = readtable(csvPath, opts);

    % Convert all text-like columns (cellstr, string, categorical) to string
    % arrays to ensure consistent downstream processing.
    for k = 1:width(T)
        col = T.(k);
        if iscellstr(col) || isstring(col) || iscategorical(col)
            T.(k) = string(col);
        end
    end
end
