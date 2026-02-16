function P = build_participant_struct(Tk)
% build_participant_struct  Construct a participant struct from event rows.
%
%   P = build_participant_struct(Tk)
%
% This function builds a single participant struct P from all event rows
% belonging to that participant (Tk). It aggregates metadata, constructs
% timeline labels, extracts door trials, trust probes, reviews,
% questionnaires, and demographics into a unified structure used
% throughout the trust modelling pipeline.
%
% High-level steps:
%   1) Extract meta-information (IDs, device, browser, etc.).
%   2) Compute within-block door order indices for door trials.
%   3) Build a timeline string array with door identifiers.
%   4) Populate P.doorTrials, including convenient door_id labels.
%   5) Build per-block timelines using door IDs.
%   6) Extract trust probes, reviews, questionnaires, and demographics.
%
% Inputs:
%   Tk - Table containing all events for a single participant. It is
%        assumed that Tk is already sorted chronologically (e.g., by
%        ts_seq in Step 1).
%
% Outputs:
%   P  - Struct holding participant-level information and derived fields:
%        - participant_id, session_id, set_id, seed, device_type, etc.
%        - timeline                : string array of event labels
%        - doorTrials              : struct array with per-door-trial data
%        - block1_timeline, ...    : block-wise timelines
%        - trustProbes             : trust probe struct(s)
%        - reviews                 : review/reputation phase struct
%        - questionnaires          : questionnaire responses
%        - demographics            : demographics struct
%
% Assumptions:
%   - Tk has at least the variables: participant_id, session_id,
%     set_id, seed, device_type, browser_name, browser_major, client_version,
%     event_type, block_index, trial_index.
%   - Door trials are labeled with event_type == "door_trial".
%   - block_index is 0-based in the raw data (converted to 1-based here).
%   - trial_index is 0-based global trial index in the raw data.

    % ---------------------------------------------------------------------
    % 1) Meta fields (first non-missing instance per column)
    % ---------------------------------------------------------------------
    P = struct();
    P.participant_id = get_string_field(Tk, "participant_id");
    P.session_id     = get_string_field(Tk, "session_id");
    P.set_id         = get_string_field(Tk, "set_id");
    P.seed           = get_numeric_field(Tk, "seed");
    P.device_type    = get_string_field(Tk, "device_type");
    P.browser_name   = get_string_field(Tk, "browser_name");
    P.browser_major  = get_numeric_field(Tk, "browser_major");
    P.client_version = get_string_field(Tk, "client_version");

    % ---------------------------------------------------------------------
    % 2) Basic event columns and block/trial indices
    % ---------------------------------------------------------------------
    et  = string(Tk.event_type);
    bi0 = get_num_col(Tk, "block_index");   % 0-based block index in raw
    ti0 = get_num_col(Tk, "trial_index");   % 0-based global trial index in raw

    % Convert to 1-based block index for struct usage.
    b1 = bi0 + 1;

    % ---------------------------------------------------------------------
    % 3) Compute within-block door_order_index for door trials
    % ---------------------------------------------------------------------
    isDoor = (et == "door_trial");
    door_order_index = nan(height(Tk), 1);

    % For each block, assign 1..N to door trials in chronological order.
    for b = 1:3
        rows_b = find(isDoor & (b1 == b));
        if ~isempty(rows_b)
            % Tk is already sorted chronologically, so this is 1..N in time.
            door_order_index(rows_b) = 1:numel(rows_b);
        end
    end

    % ---------------------------------------------------------------------
    % 4) Build map: (block_index, trial_index_raw) -> door_order_index
    % ---------------------------------------------------------------------
    % This map allows us to translate probe trial_index (raw global index)
    % into a within-block door_order_index.
    key = @(blk, tri) sprintf('%d;%d', blk, tri);  % blk: 1-based block; tri: 0-based global trial index
    K = containers.Map('KeyType', 'char', 'ValueType', 'double');

    for r = find(isDoor).'
        if ~isnan(b1(r)) && ~isnan(ti0(r)) && ~isnan(door_order_index(r))
            K(key(b1(r), ti0(r))) = door_order_index(r);
        end
    end

    % ---------------------------------------------------------------------
    % 5) Global timeline using 1-based block + within-block door_order_index
    % ---------------------------------------------------------------------
    % For door trials, the timeline stores "door_b_k". For other events, it
    % stores the event_type string.
    timeline = strings(height(Tk), 1);
    for i = 1:height(Tk)
        if et(i) == "door_trial"
            b = b1(i);
            k = door_order_index(i);
            if isnan(b) || isnan(k)
                timeline(i) = "door_?_?";
            else
                timeline(i) = "door_" + string(b) + "_" + string(k);
            end
        else
            timeline(i) = et(i);
        end
    end
    P.timeline = timeline;

    % ---------------------------------------------------------------------
    % 6) Door trials with door_order_index and convenience IDs
    % ---------------------------------------------------------------------
    P.doorTrials = extract_door_trials_with_order(Tk, door_order_index);

    % Add door_id and preserve trial_index_raw explicitly.
    for i = 1:numel(P.doorTrials)
        if ~isnan(P.doorTrials(i).block_index) && ~isnan(P.doorTrials(i).trial_index)
            P.doorTrials(i).door_id = "door_" + string(P.doorTrials(i).block_index) + "_" + string(P.doorTrials(i).trial_index);
        else
            P.doorTrials(i).door_id = "";
        end
        % Preserve the raw global trial index as a dedicated field.
        P.doorTrials(i).trial_index_raw = P.doorTrials(i).trial_index_raw; %#ok<*NASGU>
    end

    % ---------------------------------------------------------------------
    % 7) Block-wise timelines using door_order_index IDs
    % ---------------------------------------------------------------------
    % For each block b, build P.blockb_timeline as a string array of event
    % labels (door identifiers or event_type names).
    for b = 1:3
        mask = (b1 == b);
        idxs = find(mask);
        seq  = strings(numel(idxs), 1);
        for j = 1:numel(idxs)
            r = idxs(j);
            if et(r) == "door_trial"
                k = door_order_index(r);
                if isnan(k)
                    seq(j) = "door_?_?";
                else
                    seq(j) = "door_" + string(b) + "_" + string(k);
                end
            else
                seq(j) = et(r);
            end
        end
        P.("block" + b + "_timeline") = seq;
    end

    % ---------------------------------------------------------------------
    % 8) Trust probes (linked to door trials via door_order_index)
    % ---------------------------------------------------------------------
    P.trustProbes = extract_trust_probes_linked(Tk, door_order_index);

    % ---------------------------------------------------------------------
    % 9) Reviews / reputation phase
    % ---------------------------------------------------------------------
    P.reviews = extract_reviews(Tk);

    % ---------------------------------------------------------------------
    % 10) Questionnaires
    % ---------------------------------------------------------------------
    P.questionnaires = extract_questionnaires(Tk);

    % ---------------------------------------------------------------------
    % 11) Demographics
    % ---------------------------------------------------------------------
    P.demographics = extract_demographics(Tk);
end

% =========================================================================
% Local numeric helper functions
% =========================================================================

function x = get_num_col(T, name)
    % GET_NUM_COL  Retrieve numeric column, coercing to double when needed.
    %
    % If the named column exists, attempt to convert it to a numeric vector.
    % Otherwise, return a NaN column of appropriate height.
    if ismember(name, T.Properties.VariableNames)
        x = T.(name);
        if iscell(x),   x = cellfun(@local_num, x); end
        if isstring(x), x = str2double(x);          end
        if ~isnumeric(x), x = double(x);            end
    else
        x = nan(height(T), 1);
    end
end

function v = local_num(a)
    % LOCAL_NUM  Convert a scalar of mixed type to a numeric value.
    if isnumeric(a)
        v = a;
    elseif isstring(a) || ischar(a)
        v = str2double(string(a));
    elseif islogical(a)
        v = double(a);
    else
        v = NaN;
    end
end
