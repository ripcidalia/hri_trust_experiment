function name = paramDisplayName(param)
% paramDisplayName Convert internal parameter identifiers to display labels.
%
% This helper maps parameter identifiers used in behavioral or trust model
% reporting to thesis-ready labels for figures and tables. Labels may include
% LaTeX math notation where appropriate.
%
% INPUTS
%   param (string|char)
%       Internal parameter identifier (e.g., "param_sd_k", "param_sd_beta").
%
% OUTPUTS
%   name (string)
%       Display-friendly label suitable for plotting. May include LaTeX
%       formatting (e.g., "$\sigma(k)$").
%
% NOTES
%   - Falls back to the original identifier if no mapping is defined.
%   - Ensures the returned label is never empty.
%

    param = string(param);
    
    switch param
        case "param_sd_k"
            name = "$\sigma(k)$";
        case "param_sd_beta"
            name = "$\sigma(\alpha)$";
        case "param_sd_eps"
            name = "$\sigma(\varepsilon)$";
        otherwise
            % Fallback: use the internal identifier directly.
            name = param;
    end
    
    % Ensure clean, non-empty label.
    name = strip(name);
    if ismissing(name) || strlength(name) == 0
        name = "UNKNOWN";
    end
end
