function [t, y, kind, w] = extract_measurements(P, dt, weights)
% extract_measurements  Extract measurement vectors from participant (NO SIMULATION).
%
% Usage:
%   [t,y,kind,w] = extract_measurements(P, dt, weights)
%
% Behavior:
%   - Uses get_cached_time_grid_and_events(P,dt) if available, otherwise build_time_grid_and_events(P,dt)
%   - Pulls out measurements(m).t, measurements(m).y, measurements(m).kind
%   - Computes per-sample weight using utils/weight_for_kind(kind,weights)
%   - Filters to finite t,y,w and w>0
%
% Outputs:
%   t    [N x 1] double
%   y    [N x 1] double
%   kind [N x 1] string
%   w    [N x 1] double

    t = zeros(0,1);
    y = zeros(0,1);
    kind = string.empty(0,1);
    w = zeros(0,1);

    measurements = [];

    try
        if exist("get_cached_time_grid_and_events", "file") == 2
            [~, ~, ~, ~, measurements] = get_cached_time_grid_and_events(P, dt);
        else
            [~, ~, ~, ~, measurements] = build_time_grid_and_events(P, dt);
        end
    catch
        measurements = [];
    end

    if isempty(measurements)
        return;
    end

    M = numel(measurements);

    tt = NaN(M,1);
    yy = NaN(M,1);
    kk = strings(M,1);
    ww = NaN(M,1);

    for m = 1:M
        % Be robust to missing fields
        if isfield(measurements(m), "t"),    tt(m) = double(measurements(m).t);    end
        if isfield(measurements(m), "y"),    yy(m) = double(measurements(m).y);    end
        if isfield(measurements(m), "kind"), kk(m) = string(measurements(m).kind); end

        % Weight only if kind was found
        if strlength(kk(m)) > 0
            ww(m) = weight_for_kind(kk(m), weights);
        end
    end

    ok = isfinite(tt) & isfinite(yy) & isfinite(ww) & (ww > 0);

    t = tt(ok);
    y = yy(ok);
    kind = kk(ok);
    w = ww(ok);
end
