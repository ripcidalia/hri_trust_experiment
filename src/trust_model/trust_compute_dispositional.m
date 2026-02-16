function tau_disp = trust_compute_dispositional(P)
% trust_compute_dispositional  Compute dispositional trust for a participant.
%
%   tau_disp = trust_compute_dispositional(P)
%
% For model identification, dispositional trust is taken directly
% from the pre 40-item questionnaire score, mapped from [0,100] to [0,1]:
%
%   tau_disp = t40_pre.total_percent / 100
%
% This is constant throughout the simulation and participant-specific.

    % Default fallback (should rarely be used if validation is correct)
    tau_disp = 0.5;

    try
        if isfield(P, "questionnaires") && ...
           isfield(P.questionnaires, "t40_pre") && ...
           isfield(P.questionnaires.t40_pre, "total_percent")

            val = P.questionnaires.t40_pre.total_percent;
            % ensure numeric
            if iscell(val), val = val{1}; end
            if isstring(val) || ischar(val)
                val = str2double(val);
            end

            if isnumeric(val) && isscalar(val) && ~isnan(val)
                tau_disp = val / 100;   % map 0–100 → 0–1
            else
                warning("trust_compute_dispositional: invalid t40_pre.total_percent, using fallback 0.5.");
            end
        else
            warning("trust_compute_dispositional: missing t40_pre.total_percent, using fallback 0.5.");
        end
    catch
        warning("trust_compute_dispositional: error reading t40_pre, using fallback 0.5.");
    end

    % Clip to [0,1]
    tau_disp = trust_clip(tau_disp);
end
