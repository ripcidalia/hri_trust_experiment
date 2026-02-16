function S = harmonize_struct_fields(S)
% harmonize_struct_fields  Ensure all structs in an array share the same fields.
%
%   S_out = harmonize_struct_fields(S_in)
%
% This utility function takes an array of structs and ensures that every
% element has the same set of field names. For any field that is missing in
% a particular struct, a sensible empty default is added based on the
% exemplar value found in other elements of the array.
%
% This is primarily used in the preprocessing pipeline to guarantee that
% participant-level structs have a consistent schema, even if certain
% fields are absent for some participants (e.g., missing demographics or
% emergency information).
%
% Inputs:
%   S - Struct array. May have heterogeneous field sets across elements.
%
% Outputs:
%   S - Struct array where each element has the same fieldnames. Missing
%       fields are filled with type-appropriate empty values inferred from
%       other elements.
%
% Behaviour:
%   1) Compute the union of all fieldnames across S.
%   2) For each struct S(k), identify which fields from the union are
%      missing.
%   3) For each missing field, search other elements for an exemplar value
%      to infer the appropriate empty type/shape.
%   4) Add the missing field with an "empty-like" value matching the
%      exemplar (via local_empty_like).

    if isempty(S)
        return;
    end

    % ---------------------------------------------------------------------
    % 1) Compute union of all field names across the struct array
    % ---------------------------------------------------------------------
    allNames = {};
    for k = 1:numel(S)
        allNames = union(allNames, fieldnames(S(k))');
    end

    % ---------------------------------------------------------------------
    % 2) For each struct, add missing fields with inferred empty defaults
    % ---------------------------------------------------------------------
    for k = 1:numel(S)
        fn = fieldnames(S(k));
        missing = setdiff(allNames, fn);

        for m = 1:numel(missing)
            f = missing{m};

            % Find an exemplar value for this field from any other struct.
            exemplar = [];
            for j = 1:numel(S)
                if isfield(S(j), f)
                    exemplar = S(j).(f);
                    if ~isempty(exemplar)
                        break;
                    end
                end
            end

            % Use the exemplar to create a type/shape-consistent empty.
            S(k).(f) = local_empty_like(exemplar);
        end
    end
end

% -------------------------------------------------------------------------
% Local helper: create an "empty-like" value from an exemplar
% -------------------------------------------------------------------------

function v = local_empty_like(exemplar)
% LOCAL_EMPTY_LIKE  Create an empty value matching exemplar's type/shape.
%
% This helper tries to return a value that is:
%   - Empty in content, but
%   - Consistent in type (and, where appropriate, shape) with the exemplar.

    if isempty(exemplar)
        v = [];
        return;
    end
    if isstring(exemplar)
        v = strings(0,1);
        return;
    end
    if ischar(exemplar)
        v = "";
        return;
    end
    if isnumeric(exemplar)
        v = [];
        return;
    end
    if islogical(exemplar)
        v = false(0,1);
        return;
    end
    if isstruct(exemplar)
        % Create an empty struct with the same fields as the exemplar.
        f = fieldnames(exemplar);
        template = cell2struct(repmat({[]}, numel(f), 1), f, 1);
        v = repmat(template, 0, 1);
        return;
    end
    if iscell(exemplar)
        v = {};
        return;
    end

    % Fallback for unrecognized types.
    v = [];
end
