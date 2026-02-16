function name = methodDisplayName(method)
% methodDisplayName Convert internal method identifiers to display names.
%
% This utility maps internal method IDs used in the codebase and results
% tables to human-readable names for figures, tables, and reports. If an
% identifier is not recognized, the original string is returned as a safe
% fallback so that labels are never empty.
%
% INPUTS
%   method (string|char)
%       Internal method identifier (e.g., "simple_selected", "optimo_lite").
%
% OUTPUTS
%   name (string)
%       Display-friendly method name suitable for plots and publications.
%
% NOTES
%   - Ensures the output is never empty.
%   - Falls back to the original identifier if no mapping is defined.
%

    method = string(method);
    
    switch method
        case "simple_selected"
            name = "Proposed model";
        case "optimo_lite"
            name = "OPTIMo-lite";
        case "optimo_lite_outcome_only"
            name = "OPTIMo-lite (outcome-only)";
        case "bump_asymmetric"
            name = "Asymmetric bump";
        case "bump_symmetric"
            name = "Symmetric bump";
    
        case "const_dispositional"
            name = "Dispositional constant";
        case "const_global_train_mean"
            name = "Global training mean";
        case "const_oracle_participant_mean"
            name = "Oracle participant mean";
    
        otherwise
            % Fallback: use the internal identifier directly.
            name = method;
    end
    
    % Ensure clean, non-empty output.
    name = strip(name);
    if ismissing(name) || strlength(name) == 0
        name = "UNKNOWN";
    end
end
