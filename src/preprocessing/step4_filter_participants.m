function step4_filter_participants(step2MatPath, valMatPath, filterOpts)
% step4_filter_participants  Select participants with valid data and optional filters.
%
%   step4_filter_participants()
%   step4_filter_participants(step2MatPath, valMatPath)
%   step4_filter_participants(step2MatPath, valMatPath, filterOpts)
%
% This function implements Step 4 of the preprocessing pipeline. It takes:
%   - The per-participant structs produced in Step 2, and
%   - The validation report produced in Step 3,
% and returns (via file output) a subset of participants that:
%   1) Pass all required validation checks, and
%   2) Optionally satisfy user-specified demographic and experimental
%      filters (gender, age_range, set_id, review_condition, device_type,
%      emergency_choice).
%
% The result is saved as:
%   derived/participants_clean_step4.mat
% which contains:
%   participants_clean      : filtered participant array
%   info                    : struct with counts and metadata
%   allOk                   : final logical mask (validation + filters)
%   allOkValidationOnly     : logical mask of validation-only pass
%   V                       : validation struct array from Step 3
%
% Usage (no extra filters, original behaviour):
%   step4_filter_participants("derived/participants_step2.mat", ...
%                             "derived/validation_report_step3.mat")
%
% Usage (example with filters):
%   opts = struct();
%   opts.gender                = "woman";
%   opts.gender_mode           = "include";
%   opts.age_range             = ["18-24","25-34"];
%   opts.age_range_mode        = "include";
%   opts.set_id                = ["A","B","C"];
%   opts.set_id_mode           = "include";
%   opts.review_condition      = "positive";
%   opts.review_condition_mode = "include";
%   opts.device_type           = "desktop";
%   opts.device_type_mode      = "include";
%   opts.emergency_choice      = "self";
%   opts.emergency_choice_mode = "include";
%
% -------------------------------------------------------------------------
% Allowed strings for filters (based on current experiment design):
%
%   gender (P.demographics.gender):
%       "woman"
%       "man"
%       "non-binary"
%       "prefer_not_to_say"
%       "self_describe"   % in this case, gender_self_desc was used at logging
%
%   age_range (P.demographics.age_range):
%       "18-24"
%       "25-34"
%       "35-44"
%       "45-54"
%       "55-64"
%       "65+"
%       "prefer_not_to_say"
%
%   set_id (P.set_id):
%       "SetA"
%       "SetB"
%       "SetC"
%       "SetD"
%       "SetE"
%       "SetF"
%       "SetG"
%       "SetH"
%
%   review_condition (P.reviews.review_condition):
%       "very_negative"
%       "moderately_negative"
%       "slightly_negative"
%       "mixed"
%       "slightly_positive"
%       "moderately_positive"
%       "very_positive"
%
%   device_type (P.device_type):
%       "desktop"
%       "mobile"
%       "tablet"
%
%   emergency_choice (P.emergency.choice):
%       "self"
%       "robot"
%
% For each filter, a corresponding "<field>_mode" can be specified:
%   "<field>_mode" = "include" (default) or "exclude".
% -------------------------------------------------------------------------
%
% Inputs (optional):
%   step2MatPath - Path to Step 2 output (participants_step2.mat).
%                  Default: "derived/participants_step2.mat".
%
%   valMatPath   - Path to Step 3 validation report (validation_report_step3.mat).
%                  Default: "derived/validation_report_step3.mat".
%
%   filterOpts   - Struct with zero or more of the filter fields described
%                  above, plus their *_mode variants. If not provided,
%                  only validation-based filtering is applied.
%
% Outputs:
%   None. Results are saved to derived/participants_clean_step4.mat and
%   progress is printed to the console.

    % ---------------------------------------------------------------------
    % 0) Handle optional inputs and basic file checks
    % ---------------------------------------------------------------------
    if nargin < 1 || isempty(step2MatPath)
        step2MatPath = "derived/participants_step2.mat";
    end
    if nargin < 2 || isempty(valMatPath)
        valMatPath = "derived/validation_report_step3.mat";
    end
    if nargin < 3
        filterOpts = struct();  % no extra filters by default
    end

    if ~isfile(step2MatPath)
        error("Step 2 file not found: %s", step2MatPath);
    end
    if ~isfile(valMatPath)
        error("Step 3 validation file not found: %s", valMatPath);
    end

    % Load participants and validation report
    S = load(step2MatPath, "participants");
    if ~isfield(S, "participants")
        error("File %s does not contain 'participants'.", step2MatPath);
    end
    participants = S.participants;

    V = load(valMatPath, "V");
    if ~isfield(V, "V")
        error("File %s does not contain 'V'.", valMatPath);
    end
    V = V.V;

    % Sanity check: same length?
    if numel(participants) ~= numel(V)
        warning("participants and V have different lengths (%d vs %d). Assuming same order but check your pipeline.", ...
                numel(participants), numel(V));
    end

    N = numel(participants);

    % ---------------------------------------------------------------------
    % 1) Base mask: all validation flags must be true
    % ---------------------------------------------------------------------
    allOkValidation = arrayfun(@(r) ...
        (r.timeline_ok        && ...
         r.door_trials_ok     && ...
         r.trust_probes_ok    && ...
         r.questionnaires_ok  && ...
         r.demographics_ok    && ...
         r.alignment_ok), ...
        V);

    fprintf('[Step 4] Valid participants (all *_ok = true): %d/%d\n', sum(allOkValidation), N);

    % ---------------------------------------------------------------------
    % 2) Additional filters (start from all-true mask and AND each filter)
    % ---------------------------------------------------------------------
    filterMask = true(size(participants));
    applyMask = @(oldMask, newMask) (oldMask & newMask);

    % =====================================================
    % Demographic filters
    % =====================================================

    % Gender filter (optional, include/exclude)
    if isfield(filterOpts, "gender") && ~isempty(filterOpts.gender)
        allowedGender = string(filterOpts.gender);
        mode = get_mode(filterOpts, "gender_mode");
        genderMask = false(size(participants));
        for i = 1:N
            g = "";
            if isfield(participants(i), "demographics") && ...
               isfield(participants(i).demographics, "gender") && ...
               ~isempty(participants(i).demographics.gender)
                g = string(participants(i).demographics.gender);
            end
            isMember = any(strcmpi(g, allowedGender));
            switch mode
                case "include"
                    genderMask(i) = isMember;
                case "exclude"
                    genderMask(i) = ~isMember;
            end
        end
        fprintf('[Step 4] Gender %s filter: %d/%d participants pass.\n', mode, sum(genderMask), N);
        filterMask = applyMask(filterMask, genderMask);
    end

    % Age-range filter (optional, include/exclude)
    if isfield(filterOpts, "age_range") && ~isempty(filterOpts.age_range)
        allowedAge = string(filterOpts.age_range);
        mode = get_mode(filterOpts, "age_range_mode");
        ageMask = false(size(participants));
        for i = 1:N
            a = "";
            if isfield(participants(i), "demographics") && ...
               isfield(participants(i).demographics, "age_range") && ...
               ~isempty(participants(i).demographics.age_range)
                a = string(participants(i).demographics.age_range);
            end
            isMember = any(strcmpi(a, allowedAge));
            switch mode
                case "include"
                    ageMask(i) = isMember;
                case "exclude"
                    ageMask(i) = ~isMember;
            end
        end
        fprintf('[Step 4] Age_range %s filter: %d/%d participants pass.\n', mode, sum(ageMask), N);
        filterMask = applyMask(filterMask, ageMask);
    end

    % =====================================================
    % Experimental / device filters
    % =====================================================

    % set_id filter (optional, include/exclude)
    if isfield(filterOpts, "set_id") && ~isempty(filterOpts.set_id)
        allowedSet = string(filterOpts.set_id);
        mode = get_mode(filterOpts, "set_id_mode");
        setMask = false(size(participants));
        for i = 1:N
            s = "";
            if isfield(participants(i), "set_id") && ~isempty(participants(i).set_id)
                s = string(participants(i).set_id);
            end
            isMember = any(strcmpi(s, allowedSet));
            switch mode
                case "include"
                    setMask(i) = isMember;
                case "exclude"
                    setMask(i) = ~isMember;
            end
        end
        fprintf('[Step 4] set_id %s filter: %d/%d participants pass.\n', mode, sum(setMask), N);
        filterMask = applyMask(filterMask, setMask);
    end

    % review_condition filter (optional, include/exclude)
    if isfield(filterOpts, "review_condition") && ~isempty(filterOpts.review_condition)
        allowedRC = string(filterOpts.review_condition);
        mode = get_mode(filterOpts, "review_condition_mode");
        rcMask = false(size(participants));
        for i = 1:N
            rc = "";
            if isfield(participants(i), "reviews") && ...
               isfield(participants(i).reviews, "review_condition") && ...
               ~isempty(participants(i).reviews.review_condition)
                rc = string(participants(i).reviews.review_condition);
            end
            isMember = any(strcmpi(rc, allowedRC));
            switch mode
                case "include"
                    rcMask(i) = isMember;
                case "exclude"
                    rcMask(i) = ~isMember;
            end
        end
        fprintf('[Step 4] review_condition %s filter: %d/%d participants pass.\n', mode, sum(rcMask), N);
        filterMask = applyMask(filterMask, rcMask);
    end

    % device_type filter (optional, include/exclude)
    if isfield(filterOpts, "device_type") && ~isempty(filterOpts.device_type)
        allowedDev = string(filterOpts.device_type);
        mode = get_mode(filterOpts, "device_type_mode");
        devMask = false(size(participants));
        for i = 1:N
            d = "";
            if isfield(participants(i), "device_type") && ~isempty(participants(i).device_type)
                d = string(participants(i).device_type);
            end
            isMember = any(strcmpi(d, allowedDev));
            switch mode
                case "include"
                    devMask(i) = isMember;
                case "exclude"
                    devMask(i) = ~isMember;
            end
        end
        fprintf('[Step 4] device_type %s filter: %d/%d participants pass.\n', mode, sum(devMask), N);
        filterMask = applyMask(filterMask, devMask);
    end

    % emergency_choice filter (optional, include/exclude)
    if isfield(filterOpts, "emergency_choice") && ~isempty(filterOpts.emergency_choice)
        allowedChoice = string(filterOpts.emergency_choice);
        mode = get_mode(filterOpts, "emergency_choice_mode");

        emergMask = false(size(participants));

        for i = 1:N
            ec = "";
            hasChoice = false;

            if isfield(participants(i), "emergency") && ~isempty(participants(i).emergency)
                E = participants(i).emergency;

                if isfield(E, "has_response") && ~isempty(E.has_response)
                    hasChoice = logical(E.has_response);
                end

                if hasChoice && isfield(E, "choice") && ~isempty(E.choice)
                    ec = string(E.choice);
                end
            end

            % Determine if this participant's emergency choice is in the allowed set.
            isMember = any(strcmpi(ec, allowedChoice));

            switch mode
                case "include"
                    % Include only those who have a choice and match it.
                    emergMask(i) = hasChoice && isMember;

                case "exclude"
                    % Exclude only if they have a choice and it matches;
                    % participants with no choice are kept.
                    if ~hasChoice
                        emergMask(i) = true;  % keep
                    else
                        emergMask(i) = ~isMember;
                    end
            end
        end

        fprintf('[Step 4] emergency_choice %s filter: %d/%d participants pass.\n', ...
            mode, sum(emergMask), N);

        filterMask = applyMask(filterMask, emergMask);
    end

    % =====================================================
    % 3) Final combined mask and output
    % =====================================================
    finalMask = allOkValidation & filterMask;

    nTotal   = N;
    nKeep    = sum(finalMask);
    nDiscard = nTotal - nKeep;

    fprintf('[Step 4] Final mask (validation + all filters): %d/%d participants kept.\n', ...
            nKeep, nTotal);

    participants_clean = participants(finalMask);

    % Ensure output folder exists
    if ~isfolder("derived")
        mkdir("derived");
    end

    % Save cleaned participants and metadata
    outPath = "derived/participants_clean_step4.mat";
    info = struct();
    info.n_total      = nTotal;
    info.n_kept       = nKeep;
    info.n_discarded  = nDiscard;
    info.created      = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    info.filters_used = filterOpts;

    allOk = finalMask;               % final mask (validation + filters)
    allOkValidationOnly = allOkValidation; %#ok<NASGU>

    save(outPath, "participants_clean", "info", "allOk", "allOkValidationOnly", "V", "-v7.3");

    fprintf('[Step 4] Filtered participants: %d total, %d kept, %d discarded.\n', ...
            nTotal, nKeep, nDiscard);

    if ~isempty(fieldnames(filterOpts))
        disp('[Step 4] Demographic/experimental filters applied:');
        disp(filterOpts);
    else
        disp('[Step 4] No demographic/experimental filters applied (all valid participants kept).');
    end

    if nDiscard > 0
        discardedIds = arrayfun(@(p) string(p.participant_id), participants(~finalMask), 'UniformOutput', true);
        fprintf('[Step 4] Discarded participant_id(s): %s\n', strjoin(discardedIds, ", "));
    end
end

% ---------------------------------------------------------
% Local helper: get filter mode ("include" or "exclude")
% ---------------------------------------------------------
function mode = get_mode(filterOpts, fieldName)
    % GET_MODE  Retrieve filter mode ('include' or 'exclude') from options.
    %
    % If fieldName is not present in filterOpts or is empty, 'include' is
    % used as the default. If provided, the value must be either
    % 'include' or 'exclude' (case-insensitive); otherwise an error is
    % thrown.

    if isfield(filterOpts, fieldName) && ~isempty(filterOpts.(fieldName))
        mode = lower(string(filterOpts.(fieldName)));
        if mode ~= "include" && mode ~= "exclude"
            error("Invalid mode '%s' for %s. Use 'include' or 'exclude'.", ...
                  mode, fieldName);
        end
    else
        mode = "include";  % default
    end
end
