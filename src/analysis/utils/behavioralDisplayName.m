function names = behavioralDisplayName(methods)
% behavioralDisplayName Convert internal behavioral model IDs to display names.
%
% This helper maps behavioral model identifiers used in the reliance/
% decision model to human-readable names for use in figures, tables, and
% thesis text. Unknown identifiers fall back to their original string values
% so that labels are never empty.
%
% INPUTS
%   methods (string array | cellstr | char array)
%       Collection of internal behavioral model identifiers, e.g.:
%         "model0_trust_as_probability"
%         "model1_threshold"
%         "model2_offset_lapse"
%
% OUTPUTS
%   names (cell array of char)
%       Display-friendly names corresponding to each identifier. Returned as
%       cellstr for compatibility with plotting functions that expect cell
%       arrays of character vectors.
%
% NOTES
%   - Ensures all returned labels are non-empty.
%   - Preserves one-to-one ordering with the input list.
%

    % Preallocate string array for display names (maximum expected length: 3).
    names = strings([3,1]);

    for i = 1:length(methods)
        method = string(methods(i));
        
        switch method
            case "model0_trust_as_probability"
                name = "Trust-as-probability";
            case "model1_threshold"
                name = "Thresholded reliance";
            case "model2_offset_lapse"
                name = "Threshold + modulation + lapse";
            otherwise
                % Fallback: use the internal identifier directly.
                name = method;
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
