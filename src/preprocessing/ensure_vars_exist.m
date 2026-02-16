function T = ensure_vars_exist(T, specs)
% ensure_vars_exist  Ensure table variables exist with appropriate types.
%
%   T = ensure_vars_exist(T, specs)
%
% This helper function ensures that a table T contains a set of required
% variables, creating any missing ones with default values that match the
% *type* (not necessarily the value) of exemplar defaults provided in
% specs. This is useful in preprocessing pipelines where some variables
% may be absent in certain datasets but are expected downstream.
%
% Inputs:
%   T     - Input table to be augmented with any missing variables.
%
%   specs - Cell array of size N x 2, where each row is:
%             { varName, exemplarDefaultValue }
%           The exemplarDefaultValue is used to determine the desired
%           storage type (string, double, logical, etc.) of the column.
%           The created column will match the height of T and be filled
%           with appropriate default values (e.g., NaN for numeric, ""
%           for string, false for logical).
%
% Outputs:
%   T     - Table with all specified variables present. Existing variables
%           are left unchanged; only missing variables are created.
%
% Behaviour:
%   - For non-empty tables (height(T) > 0), missing variables are created
%     with Hx1 columns, where H = height(T).
%   - For empty tables, missing variables are created as 0x1 columns of
%     the appropriate type.
%   - The exemplar type is inferred using basic MATLAB type checks
%     (isstring, ischar, isnumeric, islogical, iscell); unknown types
%     default to cell arrays.
%
% Note:
%   - The function is type-oriented rather than value-oriented: the
%     exemplar specifies the desired type, not the value pattern.

    H = height(T);

    for i = 1:size(specs, 1)
        vn = specs{i, 1};  % variable name
        dv = specs{i, 2};  % exemplar default value

        % Skip if the variable already exists.
        if ismember(vn, T.Properties.VariableNames)
            continue;
        end

        % Determine type from exemplar and allocate a correctly sized column.
        if H == 0
            % For empty tables, add an empty column of the right type.
            T.(vn) = local_empty_like(dv);
        else
            % For non-empty tables, add a filled column of the right type.
            T.(vn) = local_filled_like(dv, H);
        end
    end
end

% -------------------------------------------------------------------------
% Local helpers for type-specific allocation
% -------------------------------------------------------------------------

function col = local_filled_like(exemplar, H)
    % LOCAL_FILLED_LIKE  Create Hx1 default column matching the exemplar's type.
    if isstring(exemplar)
        col = strings(H, 1);            % missing strings ("" as default)
    elseif ischar(exemplar)
        col = strings(H, 1);            % normalize char exemplar to string
    elseif isnumeric(exemplar)
        col = NaN(H, 1);                % NaN for numeric defaults
    elseif islogical(exemplar)
        col = false(H, 1);              % false for logical defaults
    elseif iscell(exemplar)
        col = cell(H, 1);
        col(:) = {[]};                  % cell defaults as empty arrays
    else
        % Fallback: generic cell column with empty entries
        col = cell(H, 1);
        col(:) = {[]};
    end
end

function col = local_empty_like(exemplar)
    % LOCAL_EMPTY_LIKE  Create 0x1 empty column matching the exemplar's type.
    if isstring(exemplar)
        col = strings(0, 1);
    elseif ischar(exemplar)
        col = strings(0, 1);
    elseif isnumeric(exemplar)
        col = zeros(0, 1);
    elseif islogical(exemplar)
        col = false(0, 1);
    elseif iscell(exemplar)
        col = cell(0, 1);
    else
        % Fallback: generic empty cell column
        col = cell(0, 1);
    end
end
