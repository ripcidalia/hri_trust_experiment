function T = decode_json_columns(T)
% decode_json_columns  Decode JSON strings in selected columns into structs.
%
%   T = decode_json_columns(T)
%
% This preprocessing helper decodes JSON-encoded fields in the event table
% into MATLAB structs while preserving the original raw text for
% traceability. The function produces two new variables:
%
%   - extra_struct    : struct decoded from T.extra_json (or [])
%   - response_struct : struct decoded from T.response    (or [])
%
% The original columns T.extra_json and T.response remain unchanged.
%
% Behaviour:
%   - If the table lacks extra_json or response variables, they are created
%     as empty-string columns.
%   - JSON decoding is performed using safejsondecode, which must return
%     either a struct or [] depending on decodability.
%   - Non-JSON content is passed through safejsondecode, which should
%     return [] without error.
%
% Inputs:
%   T - Event table containing textual JSON fields.
%
% Outputs:
%   T - Table with added variables extra_struct and response_struct.

    % Ensure the expected JSON columns exist.
    if ~ismember('extra_json', T.Properties.VariableNames)
        T.extra_json = strings(height(T), 1);
    end
    if ~ismember('response', T.Properties.VariableNames)
        T.response = strings(height(T), 1);
    end

    % Allocate output cell arrays for decoded JSON content.
    extra_struct    = cell(height(T), 1);
    response_struct = cell(height(T), 1);

    % Decode per row using safejsondecode, which handles invalid JSON safely.
    for i = 1:height(T)
        extra_struct{i}    = safejsondecode(T.extra_json(i));
        response_struct{i} = safejsondecode(T.response(i));
    end

    % Attach decoded content to the table.
    T.extra_struct    = extra_struct;
    T.response_struct = response_struct;
end
