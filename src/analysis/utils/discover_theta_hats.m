function thetaList = discover_theta_hats(R)
% discover_theta_hats  Discover fitted parameter vectors in a results struct.
%
% This utility scans a loaded MAT-file struct (typically produced by a fitting
% routine) and extracts all parameter vectors whose variable names follow the
% convention:
%
%     theta_hat_<optimizer_name>
%
% The function is intentionally generic and does not assume any fixed set of
% optimizers. It is used by Steps A2–A5 to automatically enumerate and compare
% multiple fitted solutions without hard-coding their names.
%
% Input
%   R (struct)
%       Struct obtained from `load(resultsMatPath)`, expected to contain one or
%       more numeric variables named `theta_hat_*`.
%
% Output
%   thetaList (table) with variables:
%       name  (string)
%           Clean optimizer identifier derived from the variable name, e.g.:
%             "theta_hat_ga_02"  ->  "ga_02"
%
%       theta (cell)
%           Column vectors of fitted parameters (each entry is v(:)).
%
% Notes
%   - Only numeric, non-empty variables are retained.
%   - If no matching variables are found, an empty table with the correct
%     variable names is returned (height = 0).
%   - The returned table is suitable for iteration and joining with
%     optimizer-comparison outputs in later analysis steps.

    % List all fields in the struct and select those matching "theta_hat_*".
    fn   = fieldnames(R);
    mask = startsWith(string(fn), "theta_hat_");

    names = string(fn(mask));

    % Return a well-formed empty table if nothing is found.
    if isempty(names)
        thetaList = table(string.empty(0,1), cell(0,1), ...
            'VariableNames', {'name','theta'});
        return;
    end

    % Preallocate containers.
    thetaCell  = cell(numel(names),1);
    cleanNames = strings(numel(names),1);

    for i = 1:numel(names)
        rawName = names(i);
        v = R.(rawName);

        % Skip non-numeric or empty entries (defensive programming).
        if ~isnumeric(v) || isempty(v)
            continue;
        end

        % Always store parameters as a column vector.
        thetaCell{i} = v(:);

        % Strip the "theta_hat_" prefix to obtain a readable optimizer label.
        % Example: "theta_hat_ga_02" -> "ga_02"
        cleanNames(i) = erase(rawName, "theta_hat_");
    end

    % Remove any entries that were skipped above.
    keep = ~cellfun(@isempty, thetaCell) & (cleanNames ~= "");
    cleanNames = cleanNames(keep);
    thetaCell  = thetaCell(keep);

    % Assemble output table.
    thetaList = table(cleanNames, thetaCell, ...
        'VariableNames', {'name','theta'});
end
