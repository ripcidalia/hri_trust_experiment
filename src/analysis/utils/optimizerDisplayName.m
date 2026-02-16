function names = optimizerDisplayName(optimizers)
% optimizerDisplayName Convert internal optimizer identifiers to display names.
%
% This helper maps optimizer identifiers used in the fitting pipeline to
% human-readable names for use in figures, tables, and reports. Unknown
% identifiers fall back to their original string values to ensure labels
% are never empty.
%
% INPUTS
%   optimizers (string array | cellstr | char array)
%       Collection of optimizer identifiers (e.g., "stage01_ga__overnight",
%       "stage03_fmincon", "best").
%
% OUTPUTS
%   names (cell array of char)
%       Display-friendly names corresponding to each optimizer identifier.
%
% NOTES
%   - Output is returned as cellstr for compatibility with plotting functions
%     that expect cell arrays of character vectors.
%   - Ensures no output label is empty or missing.
%

    % Preallocate string array for display names (maximum expected length: 5).
    names = strings([5,1]);

    for i = 1:length(optimizers)
        optimizer = string(optimizers(i));
        
        switch optimizer
            case "best"
                name = "Best";
            case "stage01_ga__overnight"
                name = "Genetic Algorithm 1";
            case "stage02_ga__overnight"
                name = "Genetic Algorithm 2";
            case "stage03_fmincon"
                name = "fmincon";
            case "stage04_patternsearch__overnight"
                name = "pattrnsearch";
          
            otherwise
                % Fallback: use the internal identifier directly.
                name = optimizer;
        end
        
        % Ensure clean, non-empty label.
        name = strip(name);
        if ismissing(name) || strlength(name) == 0
            name = "UNKNOWN";
        end

        names(i) = name;
    end

    % Convert to cell array of character vectors for downstream compatibility.
    names = cellstr(names);
end
