function step2_build_participants(matPath)
% step2_build_participants  Build per-participant structs from event table.
%
%   step2_build_participants(matPath)
%
% This function implements Step 2 of the preprocessing pipeline. It takes
% the normalized event table saved in Step 1 and:
%   1) Checks for the presence of key identifiers.
%   2) Groups events by participant (using participant_id, with a fallback
%      to session_id where participant_id is missing).
%   3) Builds a participant struct for each group using build_participant_struct.
%   4) Extracts emergency trial information (if present) and attaches it as
%      a dedicated sub-struct P.emergency for each participant.
%   5) Harmonizes struct fields across all participants.
%   6) Saves the resulting participant array to derived/participants_step2.mat.
%
% Inputs:
%   matPath - Path to the MAT file produced by Step 1, containing a table T
%             with event-level data and required variables:
%               - participant_id
%               - session_id
%               - event_type
%
% Outputs:
%   None (results are saved to disk):
%       derived/participants_step2.mat
%         -> participants : array of per-participant structs.
%
% Participant struct:
%   Each participant struct Pk is created by build_participant_struct and
%   then extended with:
%     Pk.emergency.has_response : logical flag, true if a valid emergency choice
%                                 ("self" or "robot") was found.
%     Pk.emergency.choice       : normalized emergency choice (lowercase),
%                                 or "" if not available / not valid.
%
% Assumptions:
%   - matPath refers to a MAT file with a variable T (table).
%   - build_participant_struct and harmonize_struct_fields are available on
%     the MATLAB path.
%   - Emergency trial events are labeled with event_type == "emergency_trial"
%     and the response is stored in one of the columns
%     'emergency_choice', 'response', or 'answer'.

    % ---------------------------------------------------------------------
    % 1) Load normalized event table from Step 1
    % ---------------------------------------------------------------------
    if ~isfile(matPath)
        error("Could not find '%s'. Run Step 1 first.", matPath);
    end
    S = load(matPath, "T");
    if ~isfield(S, "T")
        error("'%s' does not contain table T.", matPath);
    end
    T = S.T;

    % Basic presence checks for required identifier columns.
    must = ["participant_id","session_id","event_type"];
    miss = must(~ismember(must, string(T.Properties.VariableNames)));
    if ~isempty(miss)
        error("Missing required columns in T: %s", strjoin(miss,", "));
    end

    % ---------------------------------------------------------------------
    % 2) Determine grouping key (participant_id with session_id fallback)
    % ---------------------------------------------------------------------
    % Group primarily by participant_id. For any rows where participant_id
    % is missing or placeholder, use session_id instead.
    pid = string(T.participant_id);
    sid = string(T.session_id);
    pid(pid=="" | pid=="<missing>") = sid(pid=="" | pid=="<missing>");

    % Preserve chronological order (Step 1 already sorted by ts_seq).
    [G, pidKeys] = findgroups(pid); %#ok<NASGU>

    % Number of distinct participants.
    K = max(G);
    participants = [];  % delayed preallocation after first struct is built

    % ---------------------------------------------------------------------
    % 3) Build participant structs and extract emergency trial info
    % ---------------------------------------------------------------------
    for k = 1:K
        rows = (G == k);
        Tk_k = T(rows, :);
        Pk   = build_participant_struct(Tk_k);

        % --------------------------------------------------------------
        % Extract emergency_trial response for this participant
        % --------------------------------------------------------------
        et = string(Tk_k.event_type);
        idxE = find(et == "emergency_trial", 1, "last");  % expect 0 or 1, but be robust

        emergency = struct();
        emergency.has_response = false;
        emergency.choice       = "";   % will be "self" / "robot" (lowercase) if available

        if ~isempty(idxE)
            raw_choice = "";

            % Try to locate the column that stores the emergency answer.
            % If the actual column name differs, this block should be
            % adapted to match the dataset.
            if ismember("emergency_choice", Tk_k.Properties.VariableNames)
                raw_choice = string(Tk_k.emergency_choice(idxE));
            elseif ismember("response", Tk_k.Properties.VariableNames)
                raw_choice = string(Tk_k.response(idxE));
            elseif ismember("answer", Tk_k.Properties.VariableNames)
                raw_choice = string(Tk_k.answer(idxE));
            else
                warning("step2_build_participants: 'emergency_trial' found for participant %s, but no obvious answer column ('emergency_choice'/'response'/'answer') was found. Please adapt this code to your actual column name.", ...
                    string(Pk.participant_id));
            end

            if raw_choice ~= ""
                choice_norm = lower(strtrim(raw_choice));

                % We expect "self" or "robot" in the cleaned string.
                validChoices = ["self","robot"];
                if any(choice_norm == validChoices)
                    emergency.choice       = choice_norm;
                    emergency.has_response = true;
                else
                    % Store raw anyway for debugging, but mark as no valid response.
                    emergency.choice       = choice_norm;
                    emergency.has_response = false;
                    warning("step2_build_participants: unexpected emergency_trial choice '%s' for participant %s (expected 'self' or 'robot').", ...
                        choice_norm, string(Pk.participant_id));
                end
            end
        end

        % Attach emergency info to participant struct.
        Pk.emergency = emergency;

        % Preallocate the participant array based on the first struct,
        % ensuring that all elements share the same initial schema.
        if k == 1
            participants = repmat(Pk, K, 1);
        end
        participants(k) = Pk;
    end

    % ---------------------------------------------------------------------
    % 4) Harmonize fields across participants and save
    % ---------------------------------------------------------------------
    % Ensure all participant structs share identical fields (schema), even
    % if some fields are missing for certain participants.
    participants = harmonize_struct_fields(participants);

    if ~isfolder("derived"), mkdir("derived"); end
    save("derived/participants_step2.mat","participants","-v7.3");

    % ---------------------------------------------------------------------
    % 5) Diagnostics (first participant)
    % ---------------------------------------------------------------------
    if ~isempty(participants)
        P = participants(1);

        fprintf('\n=== Step 2 Diagnostics (first participant) ===\n');

        % Meta-information
        fprintf('\nMeta:\n');
        disp(struct( ...
            'participant_id', P.participant_id, ...
            'session_id',     P.session_id, ...
            'set_id',         P.set_id, ...
            'seed',           P.seed, ...
            'device_type',    P.device_type, ...
            'browser_name',   P.browser_name, ...
            'browser_major',  P.browser_major, ...
            'client_version', P.client_version));

        % Timeline summary
        nShow = min(10, numel(P.timeline));
        fprintf('\nFirst %d timeline entries:\n', nShow);
        disp(P.timeline(1:nShow));

        % Door trials
        fprintf('\ndoorTrials count: %d\n', numel(P.doorTrials));
        if ~isempty(P.doorTrials)
            fprintf('First doorTrial:\n');
            disp(P.doorTrials(1));
        end

        % Demographics (print if anything meaningful exists)
        if isfield(P, "demographics") && ~isempty(P.demographics)
            demo = P.demographics;
            if isfield(demo, "age_range") || isfield(demo, "gender")
                fprintf('\nDemographics:\n');
                disp(struct( ...
                    'age_range', demo.age_range, ...
                    'gender', demo.gender ));
            end
        end

        % Emergency trial diagnostics (if present)
        if isfield(P, "emergency")
            fprintf('\nEmergency trial:\n');
            disp(P.emergency);
        end

        fprintf('=============================================\n\n');
    end

    fprintf('[Step 2] Built %d participant struct(s). Saved to derived/participants_step2.mat\n', numel(participants));
end
