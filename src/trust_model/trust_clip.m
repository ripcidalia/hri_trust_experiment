function x = trust_clip(x)
% trust_clip  Saturate trust values to the [0, 1] interval.
%
%   x = trust_clip(x)
%
% In the trust model, all internal trust components and the combined trust
% state τ(t) are interpreted as probabilities / normalized scores and are
% therefore constrained to the closed interval [0,1]. This helper enforces
% that constraint element-wise.
%
% Inputs:
%   x  - numeric array (scalar, vector, or matrix) containing trust values
%        that may fall slightly outside [0,1] due to numerical integration,
%        parameterization, or optimization steps.
%
% Outputs:
%   x  - same size as input, real-valued, with all elements clipped to the
%        interval [0,1].
%
% Behaviour:
%   - Any small imaginary part (e.g., from complex arithmetic or solver
%     round-off) is discarded via real(x).
%   - Values less than 0 are set to 0.
%   - Values greater than 1 are set to 1.

    % Discard any spurious imaginary parts from numerical computations
    x = real(x);

    % Element-wise saturation to the [0,1] interval
    x(x < 0) = 0;
    x(x > 1) = 1;
end
