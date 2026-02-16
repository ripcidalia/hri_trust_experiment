function trust_debug_log(tag, info)
% trust_debug_log  Append trust-model debug information to a text log.
%
%   trust_debug_log(tag, info)
%
% This utility writes a single timestamped line of diagnostic information
% to a persistent text file in the "derived" directory. It is intended for
% internal debugging of the trust dynamics (e.g., detecting NaNs or values
% outside expected ranges) without interrupting the main parameter fitting
% or simulation runs.
%
% Inputs:
%   tag   - short string label describing the event type
%           (e.g., "tau_lat_invalid", "tau_rep_invalid").
%   info  - struct with arbitrary fields that provide context about the
%           event. The struct is JSON-encoded before being written.
%           If info is not a struct, it is wrapped into a struct with
%           field "data".
%
% Behavior:
%   - On first use in the current MATLAB session, a log file is created
%     in the "derived" directory with a timestamped name such as:
%         derived/trust_debug_log_20250212_153045.txt
%   - Each call appends a single line:
%         [YYYY-mm-dd HH:MM:SS.FFF] TAG | {json-encoded-info}
%   - If file creation fails once, logging is disabled for the remainder
%     of the session.
%   - No console output is produced unless the file cannot be opened.

    persistent logFilePath isEnabled

    % ---------------------------------------------------------------------
    % Initialize logging flag on first call (enabled by default)
    % ---------------------------------------------------------------------
    if isempty(isEnabled)
        % Set isEnabled to false manually in the debugger if you want to
        % disable logging for a session.
        isEnabled = true;
    end

    if ~isEnabled
        return;
    end

    % ---------------------------------------------------------------------
    % Normalize 'info' so jsonencode can handle it robustly
    % ---------------------------------------------------------------------
    if nargin < 2 || isempty(info)
        info = struct();
    elseif ~isstruct(info)
        % Wrap non-struct payloads into a struct
        info = struct('data', info);
    end

    % ---------------------------------------------------------------------
    % Lazily create the log file path (once per MATLAB session)
    % ---------------------------------------------------------------------
    if isempty(logFilePath)
        if ~isfolder("derived")
            mkdir("derived");
        end
        timestamp = datestr(now, "yyyymmdd_HHMMSS");
        logFilePath = fullfile("derived", ...
            sprintf("trust_debug_log_%s.txt", timestamp));
    end

    % ---------------------------------------------------------------------
    % Open file in append mode; disable logging if this fails
    % ---------------------------------------------------------------------
    fid = fopen(logFilePath, "a");
    if fid < 0
        % If the file cannot be opened, issue a warning once and then
        % disable further logging attempts.
        warning("trust_debug_log: could not open log file '%s' for append.", logFilePath);
        isEnabled = false;
        return;
    end

    % ---------------------------------------------------------------------
    % Build log entry: timestamp, tag, JSON payload
    % ---------------------------------------------------------------------
    tstr = datestr(now, "yyyy-mm-dd HH:MM:SS.FFF");

    % Encode info as JSON (available in MATLAB R2016b and later)
    try
        payload = jsonencode(info);
    catch
        % Fallback: use a simple placeholder if JSON encoding fails
        payload = "<jsonencode_failed>";
    end

    % One log line per event
    fprintf(fid, "[%s] %s | %s\n", tstr, char(tag), payload);

    fclose(fid);
end
