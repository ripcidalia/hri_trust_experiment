function theta = find_theta_in_struct(S)
% find_theta_in_struct  Find a theta vector inside a loaded .mat struct.
%
% Usage:
%   S = load(matPath);
%   theta = find_theta_in_struct(S);
%
% Behavior:
%   - First checks common field names (theta_star, selected_theta, etc.).
%   - If not found, falls back to scanning all fields and returning the
%     first numeric vector with length >= 2.
%   - Returns [] if nothing plausible is found.

    theta = [];

    if ~isstruct(S)
        return;
    end

    % Common names used across steps / legacy code
    cand = ["theta_selected","theta_star","theta","selected_theta","theta_hat"];

    for i = 1:numel(cand)
        f = cand(i);
        if isfield(S, f)
            v = S.(f);
            if isnumeric(v) && ~isempty(v) && isvector(v)
                theta = v(:);
                return;
            end
        end
    end

    % Fallback: scan all fields
    fn = fieldnames(S);
    for k = 1:numel(fn)
        v = S.(fn{k});
        if isnumeric(v) && ~isempty(v) && isvector(v) && numel(v) >= 2
            theta = v(:);
            return;
        end
    end
end
