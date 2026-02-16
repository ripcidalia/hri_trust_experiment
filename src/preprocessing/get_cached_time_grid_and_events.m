function [t_grid, doors_at_k, doorEvents, meas_at_k, measurements] = ...
         get_cached_time_grid_and_events(P, dt)
% get_cached_time_grid_and_events  Memoized wrapper around build_time_grid_and_events.
%
% This helper caches the time grid, door-event mapping, and measurement
% bookkeeping for each participant and time step dt. These structures depend
% only on participant data (P) and dt, and do not depend on theta, so it is
% safe to reuse them across objective evaluations during optimisation.
%
% Notes on parallel execution:
%   - In Parallel Computing Toolbox, each worker has its own persistent
%     workspace. The cache is therefore worker-local and thread-safe.
%   - This avoids recomputing grid/event bookkeeping repeatedly within each
%     worker during GA/patternsearch runs.
%
% Cache key:
%   - participant_id + dt (formatted with fixed precision)
%   - If participant_id is missing, fall back to a hash-like key from sizes.

    persistent cacheMap cacheHits cacheMisses
    if isempty(cacheMap)
        cacheMap    = containers.Map('KeyType', 'char', 'ValueType', 'any');
        cacheHits   = 0;
        cacheMisses = 0;
    end

    % ---- Build a stable cache key ----
    if isfield(P, "participant_id") && ~isempty(P.participant_id)
        pid = string(P.participant_id);
    else
        % Fallback: use structural signature (not perfect, but avoids crashes)
        nDoors  = 0; nProbes = 0;
        if isfield(P, "doorTrials") && ~isempty(P.doorTrials),  nDoors  = numel(P.doorTrials);  end
        if isfield(P, "trustProbes") && ~isempty(P.trustProbes), nProbes = numel(P.trustProbes); end
        pid = sprintf("unknownPID_doors%d_probes%d", nDoors, nProbes);
    end

    % Fixed formatting reduces key drift due to floating-point representation
    dt_key = sprintf("%.6f", dt);
    key    = char(pid + "_" + dt_key);

    % ---- Return cached entry if present ----
    if isKey(cacheMap, key)
        entry = cacheMap(key);
        cacheHits = cacheHits + 1;

        t_grid       = entry.t_grid;
        doors_at_k   = entry.doors_at_k;
        doorEvents   = entry.doorEvents;
        meas_at_k    = entry.meas_at_k;
        measurements = entry.measurements;
        return;
    end

    % ---- Cache miss: build and store ----
    cacheMisses = cacheMisses + 1;

    [t_grid, doors_at_k, doorEvents, meas_at_k, measurements] = ...
        build_time_grid_and_events(P, dt);

    entry = struct();
    entry.t_grid       = t_grid;
    entry.doors_at_k   = doors_at_k;
    entry.doorEvents   = doorEvents;
    entry.meas_at_k    = meas_at_k;
    entry.measurements = measurements;

    cacheMap(key) = entry;

   % Optional lightweight diagnostic (disabled by default):
   % Uncomment to verify caching behaviour.
   % if mod(cacheMisses, 50) == 0
   %    fprintf("[grid-cache] misses=%d, hits=%d, size=%d\n", ...
   %            cacheMisses, cacheHits, cacheMap.Count);
   % end
end