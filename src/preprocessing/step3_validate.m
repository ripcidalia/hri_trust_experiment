function step3_validate(matPath)
% step3_validate  Validate participant structs and generate a summary report.
%
%   step3_validate(matPath)
%
% This function implements Step 3 of the preprocessing pipeline. It loads
% the per-participant structs produced in Step 2 and runs a series of
% validation checks on each participant, using validate_participant.
% The results are:
%   - Printed as a concise per-participant summary to the console.
%   - Saved as a MAT file (V: array of validation structs).
%   - Exported as a CSV table for quick inspection.
%
% The validation covers:
%   - Timeline consistency.
%   - Door trial extraction and integrity.
%   - Trust probe extraction and alignment.
%   - Questionnaire extraction.
%   - Demographics extraction.
%   - Emergency trial extraction.
%   - Block-wise door trial counts and alignment.
%
% Inputs:
%   matPath - Path to the MAT file produced by Step 2, expected to contain
%             a variable 'participants' (array of participant structs).
%
% Outputs:
%   None. The function saves:
%       derived/validation_report_step3.mat  (struct array V)
%       derived/validation_report_step3.csv  (flat table with key fields)
%
% Assumptions:
%   - validate_participant.m is available on the MATLAB path and returns a
%     struct with fields such as:
%       timeline_ok, door_trials_ok, trust_probes_ok, questionnaires_ok,
%       demographics_ok, emergency_ok, alignment_ok, n_door_trials,
%       n_trust_probes, block_counts, issues, participant_id.

    % ---------------------------------------------------------------------
    % 1) Load participants from Step 2 output
    % ---------------------------------------------------------------------
    if ~isfile(matPath)
        error("Not found: %s (run Step 2 first).", matPath);
    end
    S = load(matPath, "participants");
    if ~isfield(S, "participants")
        error("MAT file does not contain 'participants'.");
    end
    P = S.participants;

    if isempty(P)
        warning("No participants to validate.");
        V = struct([]); %#ok<NASGU>
        return;
    end

    fprintf('\n=== Step 3: Validation Summary ===\n');

    % ---------------------------------------------------------------------
    % 2) Validate first participant to derive schema and print summary line
    % ---------------------------------------------------------------------
    V1 = validate_participant(P(1));
    K  = numel(P);
    V  = repmat(V1, K, 1);

    % Optional header description:
    % P### | pid=... | timeline: | doors: | probes: | qnn: | demo: | emerg: | blocks:

    % First participant line
    fprintf(['P%03d | pid=%s | timeline:%s | doors:%s | probes:%s | qnn:%s | ' ...
             'demo:%s | emerg:%s | blocks:%s\n'], ...
        1, safe_str(P(1).participant_id), ...
        passfail(V1.timeline_ok), ...
        passfail(V1.door_trials_ok), ...
        passfail(V1.trust_probes_ok), ...
        passfail(V1.questionnaires_ok), ...
        passfail(V1.demographics_ok), ...
        passfail(V1.emergency_ok), ...       % emergency_ok
        passfail(V1.alignment_ok));

    if ~isempty(V1.issues)
        for b = 1:numel(V1.issues)
            fprintf('   - %s\n', V1.issues{b});
        end
    end

    % ---------------------------------------------------------------------
    % 3) Validate remaining participants and print summary lines
    % ---------------------------------------------------------------------
    for i = 2:K
        V(i) = validate_participant(P(i));
        fprintf(['P%03d | pid=%s | timeline:%s | doors:%s | probes:%s | qnn:%s | ' ...
                 'demo:%s | emerg:%s | blocks:%s\n'], ...
            i, safe_str(P(i).participant_id), ...
            passfail(V(i).timeline_ok), ...
            passfail(V(i).door_trials_ok), ...
            passfail(V(i).trust_probes_ok), ...
            passfail(V(i).questionnaires_ok), ...
            passfail(V(i).demographics_ok), ...
            passfail(V(i).emergency_ok), ...   % emergency_ok
            passfail(V(i).alignment_ok));
        if ~isempty(V(i).issues)
            for b = 1:numel(V(i).issues)
                fprintf('   - %s\n', V(i).issues{b});
            end
        end
    end
    fprintf('==================================\n\n');

    % ---------------------------------------------------------------------
    % 4) Save validation results (MAT + CSV)
    % ---------------------------------------------------------------------
    if ~isfolder("derived"), mkdir("derived"); end

    % Build a flat struct array suitable for CSV export. arrayfun returns
    % 1xK struct array; struct2table converts it to a table.
    rows = arrayfun(@flat_for_table, V);
    T = struct2table(rows);
    writetable(T, "derived/validation_report_step3.csv");
    save("derived/validation_report_step3.mat", "V", "-v7.3");

    fprintf('[Step 3] Saved:\n  - derived/validation_report_step3.csv\n  - derived/validation_report_step3.mat\n');
end

% -------------------------------------------------------------------------
% Local helper functions
% -------------------------------------------------------------------------

function s = passfail(tf)
    % PASSFAIL  Map logical/empty flags to "OK"/"FAIL"/"?" strings.
    if isempty(tf), s = "?"; return; end
    if tf, s = "OK"; else, s = "FAIL"; end
end

function s = safe_str(x)
    % SAFE_STR  Convert identifier-like values to a printable string.
    try
        s = string(x);
        if s == "", s = "<empty>"; end
    catch
        s = "<err>";
    end
end

function R = flat_for_table(v)
% FLAT_FOR_TABLE  Convert a validation struct to a flat scalar struct.
%
% This helper reshapes a single validation entry v into a scalar struct
% with basic fields suitable for CSV export. It extracts logical flags,
% counts, and block-wise door counts.

    R = struct( ...
        'participant_id',         v.participant_id, ...
        'timeline_ok',            logical(v.timeline_ok), ...
        'door_trials_ok',         logical(v.door_trials_ok), ...
        'trust_probes_ok',        logical(v.trust_probes_ok), ...
        'questionnaires_ok',      logical(v.questionnaires_ok), ...
        'demographics_ok',        logical(v.demographics_ok), ...
        'emergency_ok',           logical(v.emergency_ok), ...   % emergency_ok in CSV
        'alignment_ok',           logical(v.alignment_ok), ...
        'n_door_trials',          double(v.n_door_trials), ...
        'n_trust_probes',         double(v.n_trust_probes), ...
        'block1_door_count',      double(v.block_counts(1)), ...
        'block2_door_count',      double(v.block_counts(2)), ...
        'block3_door_count',      double(v.block_counts(3)) ...
    );
end
