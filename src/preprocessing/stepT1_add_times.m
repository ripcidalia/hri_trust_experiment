function stepT1_add_times(cleanMatPath, eventsCsvPath)
% stepT1_add_times  Enrich cleaned participants with continuous time stamps.
%
%   stepT1_add_times(cleanMatPath, eventsCsvPath)
%
% This step (T1) augments the participant structs produced by Step 4 with
% continuous time information (in seconds) derived from the raw events
% table. For each participant, it:
%
%   - Reloads the raw events table from the original CSV.
%   - Computes a relative time axis t_s (seconds) per event, using:
%       * ts_client (datetime) when available, or
%       * ts_seq as a monotone fallback.
%   - Shifts the time axis so that the first door_trial event occurs at
%     t = 10 seconds (if present).
%   - Injects time stamps into:
%       P.timeline_t_s                 (aligned with P.timeline)
%       P.doorTrials(k).t_s            (door_trial completion times)
%       P.trustProbes(j).t_s           (trust probe times)
%       P.questionnaires.t40_pre.t_s
%       P.questionnaires.t40_post.t_s  (forced to last door trial + 10s)
%       P.questionnaires.t14_mid1.t_s
%       P.questionnaires.t14_mid2.t_s
%
% Inputs:
%   cleanMatPath   - Path to MAT file with participants_clean produced by
%                    Step 4 (default:
%                    "derived/participants_clean_step4.mat").
%   eventsCsvPath  - Path to raw events CSV file (default:
%                    fullfile("rawData","events.csv")).
%
% Outputs:
%   This function saves a new MAT file:
%       derived/participants_time_stepT1.mat
%   containing:
%       participants_clean : enriched participant structs with time fields
%       info_time          : metadata about the enrichment operation.
%
% Notes:
%   - This function does not modify the structure of P.timeline; it only
%     adds a parallel numeric vector P.timeline_t_s.
%   - If a participant cannot be matched to any rows in the events table,
%     a warning is issued and that participant is skipped for time
%     enrichment.
%   - When counts of door_trial or trust_probe events differ between the
%     raw table and the structs, times are mapped up to the minimum
%     available length, with a warning for the mismatch.

    if nargin < 1 || isempty(cleanMatPath)
        cleanMatPath = "derived/participants_clean_step4.mat";
    end
    if nargin < 2 || isempty(eventsCsvPath)
        eventsCsvPath = fullfile("rawData","events.csv");
    end

    if ~isfile(cleanMatPath)
        error("participants_clean file not found: %s", cleanMatPath);
    end
    if ~isfile(eventsCsvPath)
        error("Events CSV not found: %s", eventsCsvPath);
    end

    % ------------------------------------------------------------
    % 1) Load participants_clean
    % ------------------------------------------------------------
    S = load(cleanMatPath, "participants_clean");
    if ~isfield(S, "participants_clean")
        error("File %s does not contain 'participants_clean'.", cleanMatPath);
    end
    participants = S.participants_clean;
    N = numel(participants);

    % ------------------------------------------------------------
    % 2) Load full events table
    % ------------------------------------------------------------
    fprintf('[Step T1] Reading events table from %s\n', eventsCsvPath);
    T = read_events_table(eventsCsvPath);

    % Sort chronologically by ts_seq if available.
    if ismember("ts_seq", T.Properties.VariableNames)
        T = sortrows(T, "ts_seq");
    end

    % ------------------------------------------------------------
    % 3) For each participant, compute relative times and inject
    % ------------------------------------------------------------
    for i = 1:N
        Pi = participants(i);
        pid = string(Pi.participant_id);

        % session_id might be missing/empty in rare cases, so handle both.
        sid = "";
        if isfield(Pi, "session_id") && ~isempty(Pi.session_id)
            sid = string(Pi.session_id);
        end

        % Filter T to this participant (and session if available).
        if sid == ""
            idxP = (string(T.participant_id) == pid);
        else
            idxP = (string(T.participant_id) == pid) & ...
                   (string(T.session_id)     == sid);
        end

        Tk = T(idxP, :);
        if isempty(Tk)
            warning("No rows found in events table for participant_id=%s, session_id=%s. Skipping time enrichment for this participant.", pid, sid);
            continue;
        end

        % Sort participant-specific events by ts_seq if present, otherwise
        % keep row order as in the original table.
        if ismember("ts_seq", Tk.Properties.VariableNames)
            Tk = sortrows(Tk, "ts_seq");
        end

        % --------------------------------------------------------
        % 3a) Compute relative times t_s from ts_client (fallback to ts_seq)
        % --------------------------------------------------------
        t_s = compute_relative_time_seconds(Tk);

        % --------------------------------------------------------
        % 3a.1) Shift times so that the first door_trial occurs at t = 10 s
        % --------------------------------------------------------
        et = string(Tk.event_type);
        rowsDoor = find(et == "door_trial", 1, "first");
        if ~isempty(rowsDoor)
            tDoor0 = t_s(rowsDoor);
            t_s = t_s - tDoor0 + 10;
        else
            % If no door trial exists, keep a simple reference: first event at t=0.
            t_s = t_s - t_s(1);
            warning("No door_trial found for participant %s; using first event as t=0.", pid);
        end

        % 3b) Store timeline_t_s aligned with P.timeline.
        %     We assume Step 2 built P.timeline in the same row order as Tk.
        if numel(Pi.timeline) ~= height(Tk)
            warning("timeline length (%d) does not match events rows (%d) for participant %s. Skipping timeline_t_s.", ...
                numel(Pi.timeline), height(Tk), pid);
        else
            participants(i).timeline_t_s = t_s(:);
        end

        % 3c) DoorTrials times: match order by event_type == "door_trial".
        rowsDoorAll = find(et == "door_trial");
        tLastDoor = [];
        if ~isempty(rowsDoorAll)
            tDoor = t_s(rowsDoorAll);
            tLastDoor = tDoor(end);

            if isfield(Pi, "doorTrials") && ~isempty(Pi.doorTrials)
                if numel(Pi.doorTrials) ~= numel(tDoor)
                    warning("Number of doorTrials (%d) != number of door_trial rows (%d) for participant %s. Mapping in min length only.", ...
                        numel(Pi.doorTrials), numel(tDoor), pid);
                end
                nMap = min(numel(Pi.doorTrials), numel(tDoor));
                for k = 1:nMap
                    participants(i).doorTrials(k).t_s = tDoor(k);
                end
            end
        end

        % 3d) Trust probes times: any event_type starting with "trust_probe".
        rowsProbe = find(startsWith(et, "trust_probe"));
        if ~isempty(rowsProbe)
            tProbe = t_s(rowsProbe);
            if isfield(Pi, "trustProbes") && ~isempty(Pi.trustProbes)
                if numel(Pi.trustProbes) ~= numel(tProbe)
                    warning("Number of trustProbes (%d) != number of trust_probe rows (%d) for participant %s. Mapping in min length only.", ...
                        numel(Pi.trustProbes), numel(tProbe), pid);
                end
                nMap = min(numel(Pi.trustProbes), numel(tProbe));
                for k = 1:nMap
                    participants(i).trustProbes(k).t_s = tProbe(k);
                end
            end
        end

        % 3e) Questionnaires times (treated as completion time).
        qtypes = { ...
            "questionnaire40pre",  "t40_pre";  ...
            "questionnaire40post", "t40_post"; ...
            "questionnaire14mid1", "t14_mid1"; ...
            "questionnaire14mid2", "t14_mid2"  ...
        };

        if isfield(Pi, "questionnaires") && ~isempty(Pi.questionnaires)
            for q = 1:size(qtypes,1)
                evtName   = qtypes{q,1};
                fieldName = qtypes{q,2};
                rowsQ = find(et == evtName);
                if ~isempty(rowsQ)
                    % If multiple rows exist, take the first.
                    tQ = t_s(rowsQ(1));
                    if isfield(Pi.questionnaires, fieldName)
                        participants(i).questionnaires.(fieldName).t_s = tQ;
                    end
                end
            end

            % Normalize t40_post timing relative to the last door trial.
            % This is only applied when the last door trial time is available
            % and the t40_post struct field exists.
            if ~isempty(tLastDoor) && isfield(Pi.questionnaires, "t40_post")
                participants(i).questionnaires.t40_post.t_s = tLastDoor + 10;
            end
        end
    end

    % ------------------------------------------------------------
    % 4) Save enriched participants
    % ------------------------------------------------------------
    if ~isfolder("derived")
        mkdir("derived");
    end

    % Keep the original naming convention: expose the enriched array
    % as 'participants_clean' so later steps can load it without changes.
    participants_clean = participants;

    info_time = struct();
    info_time.source_clean_file = cleanMatPath;
    info_time.events_csv        = eventsCsvPath;
    info_time.created           = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    info_time.n_participants    = numel(participants_clean);

    outPath = "derived/participants_time_stepT1.mat";
    save(outPath, "participants_clean", "info_time", "-v7.3");

    fprintf('[Step T1] Time enrichment completed for %d participants.\n', numel(participants_clean));
    fprintf('          Saved to %s\n', outPath);
end


% =================================================================
% Helper: compute_relative_time_seconds
% =================================================================
function t_s = compute_relative_time_seconds(Tk)
% compute_relative_time_seconds  Derive a relative time axis (seconds).
%
%   t_s = compute_relative_time_seconds(Tk)
%
% Computes a per-row time stamp in seconds for the table Tk, using:
%   - Tk.ts_client, parsed as UTC datetimes, when available and valid.
%   - Otherwise, Tk.ts_seq as a monotone surrogate (units arbitrary).
%   - As a last resort, the row index (0,1,2,...) if neither field exists.
%
% The times are returned relative to the first event in Tk:
%   t_s(1) = 0, and t_s(k) >= 0 for k > 1 in all fallback modes.
%
% Inputs:
%   Tk  - Sub-table of the events for one participant/session, sorted
%         chronologically by ts_seq when available.
%
% Outputs:
%   t_s - Column vector of length height(Tk) with relative times in
%         seconds (or surrogate units based on ts_seq or row index).

    n = height(Tk);
    t_s = zeros(n,1);

    % Case 1: ts_client exists and is non-empty.
    if ismember("ts_client", Tk.Properties.VariableNames)
        ts = Tk.ts_client;

        % Convert to datetime if needed.
        if iscellstr(ts) || isstring(ts) || ischar(ts)
            tsStr = string(ts);
            try
                % Expected format: 2025-11-11T13:47:25.601Z
                dt = datetime(tsStr, ...
                    "InputFormat", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", ...
                    "TimeZone", "UTC");
            catch
                try
                    % Fallback: same but without milliseconds.
                    dt = datetime(tsStr, ...
                        "InputFormat", "yyyy-MM-dd'T'HH:mm:ss'Z'", ...
                        "TimeZone", "UTC");
                catch
                    warning("Failed to parse ts_client; falling back to ts_seq for timing.");
                    dt = NaT(size(tsStr));
                end
            end
        elseif isdatetime(ts)
            dt = ts;
        else
            dt = NaT(n,1);
        end

        % If all entries are valid datetimes, compute seconds since first.
        if all(~isnat(dt))
            t0 = dt(1);
            t_s = seconds(dt - t0);
            return;
        else
            warning("ts_client is present but could not be fully parsed; falling back to ts_seq.");
        end
    end

    % Case 2: fallback to ts_seq as pseudo-time.
    if ismember("ts_seq", Tk.Properties.VariableNames)
        seq = double(Tk.ts_seq);
        t_s = seq - seq(1);
    else
        % Last resort: use row index as time.
        t_s = (0:n-1)';
    end
end
