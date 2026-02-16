function [ok, msg, nProbes] = check_trust_probes(P)
% check_trust_probes  Validate trust-probe completeness and data quality.
%
%   [ok, msg, nProbes] = check_trust_probes(P)
%
% This function performs structural and value-level validation of the
% participant’s trust probes. Trust probes are single-shot slider ratings
% administered throughout the interaction and mapped to a 0–100 scale.
%
% Validation rules:
%   - Probes must exist (non-empty).
%   - Each participant must have the expected total count (experiment design).
%   - All probe values must be finite numeric values within [0, 100].
%
% Inputs:
%   P         - Participant struct containing:
%                 P.trustProbes : 1×M struct array
%                     Each element must have field:
%                        value : numeric scalar (0–100)
%
% Outputs:
%   ok        - Logical flag indicating whether all trust-probe criteria
%               are satisfied.
%   msg       - Diagnostic message describing the first detected issue.
%               Empty if ok == true.
%   nProbes   - Number of trust probes detected for this participant.

    % Extract probes and count.
    U = P.trustProbes;
    nProbes = numel(U);

    % A participant must have at least one probe.
    if nProbes == 0
        ok  = false;
        msg = "No trust probes present.";
        return;
    end

    % Probe values must be valid numerics. Missing values (NaN) are not allowed.
    vals = arrayfun(@(x) x.value, U);
    if any(isnan(vals))
        ok  = false;
        msg = "Some trust probes have NaN values.";
        return;
    end

    % All probes must lie within the valid slider range [0, 100].
    if any(vals < 0 | vals > 100)
        bad = find(vals < 0 | vals > 100, 1);
        ok  = false;
        msg = sprintf("Trust probe #%d out of range [0,100].", bad);
        return;
    end

    % Expected total number of trust probes for this experiment design.
    % If the design changes, update expectedTotal accordingly.
    expectedTotal = 13;
    if nProbes ~= expectedTotal
        ok  = false;
        msg = sprintf("Expected %d trust probes, found %d.", expectedTotal, nProbes);
        return;
    end

    % All checks passed.
    ok  = true;
    msg = "";
end
