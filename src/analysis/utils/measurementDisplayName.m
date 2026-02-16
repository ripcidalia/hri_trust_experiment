function lbl = measurementDisplayName(kind)
% measurementDisplayName Convert internal measurement IDs to display labels.
%
% This helper maps internal measurement identifiers used in the trust
% modeling pipeline to thesis-ready labels for plots and tables. Labels are
% formatted using LaTeX math notation where appropriate (e.g., questionnaire
% notation q^(n) with contextual subscripts).
%
% INPUTS
%   kind (string|char)
%       Internal measurement identifier (e.g., "probe", "t14_mid1",
%       "t14_mid2", "t40_post").
%
% OUTPUTS
%   lbl (string)
%       Display label suitable for plotting. May include LaTeX formatting
%       (e.g., "$q^{(14)}_{\mathrm{mid1}}$").
%
% NOTES
%   - Falls back to the original identifier if no mapping is defined.
%   - Ensures the returned label is never empty.
%

    kind = string(kind);

    switch kind
        case "probe"
            lbl = "$q^{(1)}$";

        case "t14_mid1"
            lbl = "$q^{(14)}_{\mathrm{mid1}}$";

        case "t14_mid2"
            lbl = "$q^{(14)}_{\mathrm{mid2}}$";

        case "t40_post"
            lbl = "$q^{(40)}_{\mathrm{post}}$";

        otherwise
            % Fallback: use the internal identifier directly.
            lbl = kind;
    end

    % Ensure clean, non-empty label.
    lbl = strip(lbl);
    if ismissing(lbl) || strlength(lbl) == 0
        lbl = "UNKNOWN";
    end
end
