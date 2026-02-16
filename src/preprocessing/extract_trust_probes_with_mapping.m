function trustProbes = extract_trust_probes_with_mapping(Tk, doorMap)
% extract_trust_probes_with_mapping  Extract trust probes with door mapping.
%
%   trustProbes = extract_trust_probes_with_mapping(Tk, doorMap)
%
% This function extracts mid-block trust probes ("trust_probe_mid") from a
% participant-level event table and expresses their indices in both:
%   - raw global trial index (0-based, as logged), and
%   - within-block door_order_index (when a corresponding door trial can be
%     identified via doorMap).
%
% The mapping doorMap is expected to map a (block, trial_index_raw) pair to
% a within-block door_order_index:
%   key = sprintf('%d;%d', block_index_1based, trial_index_raw_0based)
%
% For each probe, the output struct contains:
%   block_index    : 1-based block index (converted from raw 0-based).
%   trial_index    : within-block door_order_index if resolvable via doorMap;
%                    NaN if no matching door trial is found.
%   trial_index_raw: raw global trial index (0-based, as in the logs).
%   value          : trust rating (0–100) if a numeric value can be
%                    extracted, otherwise NaN.
%
% Inputs:
%   Tk      - Table of events for a single participant, containing
%             event_type, block_index, trial_index, and optionally
%             response_struct, response, and extra_struct.
%   doorMap - containers.Map mapping '(block_1based;trial_index_raw_0based)'
%             keys to within-block door_order_index values.
%
% Outputs:
%   trustProbes - 1×M struct array of trust probe entries with index
%                 information and numeric values.
%
% Assumptions:
%   - Event rows for trust probes have event_type == "trust_probe_mid".
%   - block_index and trial_index columns follow the same conventions as
%     for door trials (0-based in logs).

    % Select only rows corresponding to mid-block trust probes.
    mask = (string(Tk.event_type) == "trust_probe_mid");
    U = Tk(mask, :);

    % If no trust probes exist for this participant, return an empty struct.
    if isempty(U)
        trustProbes = struct([]);
        return;
    end

    % Preallocate output struct array.
    M = height(U);
    trustProbes = repmat(struct( ...
        'block_index', NaN, ...
        'trial_index', NaN, ...       % within-block door_order_index if resolvable
        'trial_index_raw', NaN, ...   % original global trial_index (0-based)
        'value', NaN ), M, 1);

    % Populate each trust probe entry.
    for i = 1:M
        % Raw block and trial indices (0-based in logs).
        b0 = local_firstnum(U, i, "block_index");
        t0 = local_firstnum(U, i, "trial_index");

        % Store 1-based block index and raw global trial index.
        trustProbes(i).block_index     = b0 + 1;  % 1-based
        trustProbes(i).trial_index_raw = t0;

        % Map to within-block door_order_index if a door with the same raw
        % global trial index exists in the provided doorMap.
        if ~isnan(b0) && ~isnan(t0)
            kkey = sprintf('%d;%d', b0 + 1, t0);
            if isKey(doorMap, kkey)
                trustProbes(i).trial_index = doorMap(kkey);
            else
                trustProbes(i).trial_index = NaN; % not resolvable; leave as NaN
            end
        end

        % Value extraction using the same logic as other trust probe helpers:
        %   1) response_struct (decoded JSON)
        %   2) response (plain numeric string)
        %   3) extra_struct (decoded JSON)
        v = [];
        if ismember("response_struct", U.Properties.VariableNames)
            RS = U.response_struct{i};
            if isstruct(RS)
                v = pick_first_numeric(RS, ["value","trust","answer","slider","rating"]);
            end
        end
        if isempty(v) && ismember("response", U.Properties.VariableNames)
            vtry = str2double(string(U.response(i)));
            if ~isnan(vtry)
                v = vtry;
            end
        end
        if isempty(v) && ismember("extra_struct", U.Properties.VariableNames)
            XS = U.extra_struct{i};
            if isstruct(XS)
                v = pick_first_numeric(XS, ["value","trust","answer","slider","rating"]);
            end
        end
        if isempty(v)
            v = NaN;
        end
        trustProbes(i).value = v;
    end
end

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------

function v = local_firstnum(T, i, name)
    % LOCAL_FIRSTNUM  Extract a numeric value from T.(name)(i) with coercion.
    %
    % If the column does not exist, returns NaN. If the entry is numeric, it
    % is cast to double; otherwise an attempt is made to parse a numeric
    % value from a string/char representation.
    if ~ismember(name, T.Properties.VariableNames)
        v = NaN;
        return;
    end
    x = T.(name)(i);
    if isnumeric(x)
        v = double(x);
    else
        v = str2double(string(x));
    end
end

function n = pick_first_numeric(S, names)
    % PICK_FIRST_NUMERIC  Return the first numeric-like field in struct S.
    %
    % Iterates over the candidate field names in 'names' and returns the
    % first one that is numeric or can be parsed as numeric from a
    % string/char. Returns [] if none of the fields yields a numeric value.
    n = [];
    for k = 1:numel(names)
        f = names(k);
        if isfield(S, f)
            x = S.(f);
            if isnumeric(x)
                n = double(x);
                return;
            end
            if isstring(x) || ischar(x)
                xv = str2double(string(x));
                if ~isnan(xv)
                    n = xv;
                    return;
                end
            end
        end
    end
end
