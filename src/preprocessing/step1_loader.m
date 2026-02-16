function step1_loader(csvPath)
% step1_loader  Load raw event CSV and perform initial normalization.
%
%   step1_loader(csvPath)
%
% This function implements Step 1 of the preprocessing pipeline for the
% SAR trust study. It loads the raw event-level CSV exported from the
% experiment platform and performs a series of normalization and sanity
% check operations before saving the result for subsequent steps.
%
% The processing sequence is:
%   1) Read the raw events CSV into a table.
%   2) Normalize column headers (lowercase, underscores, etc.).
%   3) Decode JSON fields from 'extra_json' and 'response' into struct
%      columns while preserving the original text.
%   4) Project commonly used fields from nested structures into top-level
%      table variables (via project_common_fields).
%   5) Sort events chronologically by 'ts_seq' when available.
%   6) Print basic diagnostics about the resulting table.
%   7) Save the normalized table to a MAT file and (best effort) CSV.
%
% Inputs:
%   csvPath - Path to the raw event CSV file.
%
% Outputs:
%   None. This function saves the normalized events to:
%       derived/normalized_events_step1.mat
%       derived/normalized_events_step1.csv  (best effort; may fail if
%                                            nested structs prevent CSV export)
%
% Assumptions:
%   - csvPath points to a valid CSV file readable by read_events_table.
%   - Helper functions read_events_table, normalize_headers,
%     decode_json_columns, project_common_fields, sort_by_ts, and
%     print_step1_diagnostics are available on the MATLAB path.
%
% Notes:
%   - The MAT file is the primary source of truth. The CSV export is
%     provided only for inspection and may not include nested structures
%     cleanly.

    % Check that the input CSV exists.
    if ~isfile(csvPath)
        error("CSV not found at: %s", csvPath);
    end

    % 1) Read table from CSV with robust import options.
    T = read_events_table(csvPath);

    % 2) Normalize variable names (lowercase, underscores, etc.).
    T = normalize_headers(T);

    % 3) Decode JSON columns ('extra_json' and JSON-like 'response')
    %    into struct-typed columns while keeping the original text.
    T = decode_json_columns(T);

    % 4) Project commonly useful fields from nested structures or JSON
    %    into dedicated top-level variables.
    T = project_common_fields(T);

    % 5) Sort chronologically by ts_seq; fall back to original file order
    %    if no such column exists.
    T = sort_by_ts(T);

    % 6) Print summary diagnostics for manual inspection.
    print_step1_diagnostics(T);

    % 7) Save normalized results for inspection and downstream steps.
    if ~isfolder("derived"), mkdir("derived"); end
    save("derived/normalized_events_step1.mat", "T", "-v7.3");
    try
        writetable(T, "derived/normalized_events_step1.csv");
    catch
        % Some tables with nested structs cannot be written to CSV cleanly.
        % The MAT file remains the primary source of truth.
    end

    fprintf('\n[Step 1] Done. Saved to:\n');
    fprintf('  - derived/normalized_events_step1.mat\n');
    fprintf('  - derived/normalized_events_step1.csv (best effort)\n\n');
end
