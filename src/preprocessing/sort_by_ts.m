function T = sort_by_ts(T)
% sort_by_ts  Sort event table by timestamp sequence if available.
%
%   T = sort_by_ts(T)
%
% This function sorts a table of experimental events by the variable
% 'ts_seq' when it exists. The goal is to ensure events appear in temporal
% order during preprocessing. If the table does not contain 'ts_seq', the
% input is returned unchanged.
%
% Sorting behaviour:
%   - If T.ts_seq is numeric, sorting is direct.
%   - If T.ts_seq is non-numeric (string, cellstr, etc.), it is converted
%     to numeric using str2double. Non-convertible values become NaN.
%   - NaN timestamps (missing or unparseable) are set to +Inf to push them
%     to the end while preserving their relative order.
%
% Inputs:
%   T - Table containing event data.
%
% Outputs:
%   T - Sorted table (or unchanged if 'ts_seq' is absent).

    % Only sort if a timestamp sequence column exists.
    if ismember('ts_seq', T.Properties.VariableNames)
        ts = T.ts_seq;

        % Ensure numeric timestamps; convert if necessary.
        if ~isnumeric(ts)
            ts = str2double(string(ts));
        end

        % Missing/unparseable values become +Inf so they sort last.
        ts(isnan(ts)) = inf;

        % Sort in ascending order of timestamp.
        [~, idx] = sort(ts, 'ascend');
        T = T(idx, :);
    end
end
