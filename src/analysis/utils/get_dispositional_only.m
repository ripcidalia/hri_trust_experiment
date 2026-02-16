function T0 = get_dispositional_only(P, fallback)
% Only trust_compute_dispositional(P); if missing/non-finite -> fallback
    try
        T0 = trust_compute_dispositional(P);
    catch
        T0 = NaN;
    end
    if ~isfinite(T0)
        T0 = fallback;
    end
end