function tau_rep0 = trust_initial_reputation(params, P)
% trust_initial_reputation  Compute initial reputation trust component.
%
%   tau_rep0 = trust_initial_reputation(params, P)
%
% This function computes the initial *reputation* component of trust,
% tau_rep0, for a given participant. The value is derived from the
% reputation / review phase collected before the main interaction
% (e.g. online reviews about the robot). It is intended as an additive
% term that can decay over time in the full model.
%
% Current behaviour:
%   - Attempt to extract "expected" and "response" from the first review
%     item:
%         P.reviews.items(1).response_struct{1}.expected
%         P.reviews.items(1).response_struct{1}.response
%   - Set tau_rep0 = expected (if it is a valid numeric scalar).
%   - If that is unavailable, fall back to:
%         P.reviews.review_expected
%     or, if that is missing as well, to:
%         params.rep.tau0
%     or finally 0.0.
%
% The final result is clipped to the range [-1, 1].
%
% Inputs:
%   params  - struct of model parameters. Only params.rep.tau0 is used
%             as a fallback if no review-based value can be extracted.
%   P       - participant struct, expected to contain a "reviews" field
%             created during preprocessing:
%
%             P.reviews.items            : [N x 1 struct] (N ≈ 1 here)
%             P.reviews.review_condition : string label
%             P.reviews.review_expected  : numeric aggregate expectation
%
%             Each P.reviews.items(k) has (among other fields):
%                 .response_struct : {1x1 cell} → struct with fields:
%                       .response  (numeric in [-1,1])
%                       .expected  (numeric in [-1,1])
%
% Output:
%   tau_rep0 - initial reputation component in [-1, 1].
%
% Notes:
%   - rep_response is currently extracted but not used in the final
%     value; this is kept for potential future extensions (e.g. modelling
%     negativity bias).
%   - If validation is strict, the main code path should normally use the
%     review-based "expected" value.

    % ----------------------------------------------------------------------
    % 1) Initialise defaults and local holders
    % ----------------------------------------------------------------------
    % Default initial reputation if nothing can be read from the data.
    tau_rep0     = 0.0;

    % Local variables to hold extracted expected / response values.
    rep_expected = NaN;
    rep_response = NaN;

    % ----------------------------------------------------------------------
    % 2) Attempt to extract expected/response from review items
    % ----------------------------------------------------------------------
    try
        if isfield(P, "reviews") && ~isempty(P.reviews) && ...
           isfield(P.reviews, "items") && ~isempty(P.reviews.items)

            items = P.reviews.items;

            % Use the first review item as the reference source
            item1 = items(1);

            if isfield(item1, "response_struct") && ...
               ~isempty(item1.response_struct) && ...
               iscell(item1.response_struct)

                % response_struct is stored as a 1x1 cell containing a struct
                s = item1.response_struct{1};

                if isstruct(s)
                    if isfield(s, "expected")
                        rep_expected = s.expected;
                    end
                    if isfield(s, "response")
                        rep_response = s.response; %#ok<NASGU> (retained for future use)
                    end
                end
            end
        end
    catch ME
        warning("trust_initial_reputation: error extracting expected/response: %s", ME.message);
    end

    % ----------------------------------------------------------------------
    % 3) Map extracted value to tau_rep0, with fallbacks
    % ----------------------------------------------------------------------
    % Primary path: use "expected" from the review item, if present.
    if ~isnan(rep_expected)
        val = rep_expected;

        % Coerce to numeric scalar if needed (cell/string → double).
        if iscell(val)
            val = val{1};
        end
        if isstring(val) || ischar(val)
            val = str2double(val);
        end

        if isnumeric(val) && isscalar(val) && isfinite(val)
            tau_rep0 = val;
        else
            warning("trust_initial_reputation: expected is non-numeric, using 0.");
            tau_rep0 = 0.0;
        end
    else
        % Fallback 1: aggregated expected reputation from preprocessing
        if isfield(P, "reviews") && isfield(P.reviews, "review_expected")
            tau_rep0 = P.reviews.review_expected;

        % Fallback 2: default value specified in params.rep.tau0
        elseif isfield(params, "rep") && isfield(params.rep, "tau0")
            tau_rep0 = params.rep.tau0;

        % Fallback 3: remain at 0.0 (defined above)
        else
            tau_rep0 = 0.0;
        end
    end

    % ----------------------------------------------------------------------
    % 4) Final safety clipping to [-1, 1]
    % ----------------------------------------------------------------------
    tau_rep0 = max(min(tau_rep0, 1), -1);
end
