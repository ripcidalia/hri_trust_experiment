function runInfo = stepA1_prepare_analysis_run(run_id, varargin)
% stepA1_prepare_analysis_run  Prepare all Step A1 run artifacts for a fixed train/valid split.
%
% This function prepares a run-local analysis directory for the A-step pipeline by:
%   1) Validating required input files (train/valid splits, calibration, weights),
%   2) Applying TRAIN-derived calibrations to the VALID split (14-item and probe mappings),
%   3) Archiving all key artifacts into a run-specific folder, and
%   4) Writing a manifest (MAT + JSON) that captures provenance and file metadata.
%
% In the broader project workflow, Step A1 creates deterministic, run-scoped inputs
% for downstream model fitting (A2–A4) and evaluation/reporting (A5–A6), ensuring that
% validation outputs are not written into shared derived/ paths unless explicitly requested.
%
% Inputs
%   run_id (string/char)
%       Unique identifier for the analysis run. Used to create:
%       <OutDir>/<run_id>/stepA1_prepare_analysis/
%
% Name-Value Arguments (all optional)
%   "TrainSplitPath" (string/char)
%       Path to TRAIN participant split MAT file (expects variable participants_clean).
%       Default: "derived/participants_train_stepV.mat"
%   "ValidSplitPath" (string/char)
%       Path to VALID participant split MAT file (expects variable participants_clean).
%       Default: "derived/participants_valid_stepV.mat"
%   "TrainParticipantsProbesPath" (string/char)
%       Path to TRAIN probes-mapped archive MAT file (optional, archived if present).
%       Default: "derived/participants_probes_mapped_stepM4.mat"
%
%   "CalibPath" (string/char)
%       Path to calibration MAT mapping 14-item/single-item -> 40-item scale (expects calib struct).
%       Default: "derived/measurement_stepM1_calibration.mat"
%   "WeightsPath" (string/char)
%       Path to measurement weights MAT file (expects weights struct).
%       Default: "derived/measurement_weights.mat"
%
%   "OutDir" (string/char)
%       Base output directory for analysis runs.
%       Default: "derived/analysis_runs"
%
%   "ValidMapped14Path" (string/char)
%       Output path for VALID participants after applying 14-item mapping (Step M2).
%       Default: <stepDir>/participants_valid_mapped14_stepM2.mat
%   "ValidProbesMappedPath" (string/char)
%       Output path for VALID participants after applying probe mapping (Step M4).
%       Default: <stepDir>/participants_valid_probes_mapped_stepM4.mat
%
%   "Overwrite" (logical scalar)
%       If false, refuses to overwrite run-local output files.
%       Default: false
%   "WriteCanonicalLatest" (logical scalar)
%       If true, also writes convenience copies into derived/ (overwriting):
%         - derived/participants_valid_mapped14_stepM2.mat
%         - derived/participants_valid_probes_mapped_stepM4.mat
%       Default: false
%
% Outputs
%   runInfo (struct)
%       Manifest struct summarizing inputs, outputs, split IDs, provenance checks,
%       calibration/weights summaries, and file metadata. Saved to:
%         - <stepDir>/manifest.mat
%         - <stepDir>/manifest.json
%
% Assumptions / Dependencies
%   - Split MAT files contain variable "participants_clean" (array of participant structs).
%   - Utility functions exist on the MATLAB path (typically under src/utils):
%       must_exist_file, ensure_dir, read_participant_ids, compare_id_sets,
%       copy_into_dir, safe_copy_into_dir, file_info_struct, save_json.
%   - Mapping steps are implemented and available on the path:
%       stepM2_apply_14mapping, stepM4_apply_probe_mapping

    % ---- parse ----
    % Input parsing is kept explicit to ensure stable defaults and validation.
    p = inputParser;
    p.addRequired("run_id", @(s) isstring(s) || ischar(s));

    p.addParameter("TrainSplitPath", "derived/participants_train_stepV.mat", @(s) isstring(s) || ischar(s));
    p.addParameter("ValidSplitPath", "derived/participants_valid_stepV.mat", @(s) isstring(s) || ischar(s));
    p.addParameter("TrainParticipantsProbesPath", "derived/participants_probes_mapped_stepM4.mat", @(s) isstring(s) || ischar(s));

    p.addParameter("CalibPath",    "derived/measurement_stepM1_calibration.mat", @(s) isstring(s) || ischar(s));
    p.addParameter("WeightsPath",    "derived/measurement_weights.mat",      @(s) isstring(s) || ischar(s));

    p.addParameter("OutDir", "derived/analysis_runs", @(s) isstring(s) || ischar(s));

    p.addParameter("ValidMapped14Path", "", @(s) isstring(s) || ischar(s));
    p.addParameter("ValidProbesMappedPath", "", @(s) isstring(s) || ischar(s));

    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("WriteCanonicalLatest", false, @(x) islogical(x) && isscalar(x));

    p.parse(run_id, varargin{:});
    args = p.Results;

    run_id = string(args.run_id);

    % ---- utils availability ----
    % Best-effort attempt to add src/utils to the path for local development.
    % This is non-fatal: if utilities are already on the path, nothing changes.
    try
        thisFile = mfilename("fullpath");
        repoRoot = fileparts(thisFile); % assumes stepA1 is at repo root or similar
        utilsDir = fullfile(repoRoot, "src", "utils");
        if isfolder(utilsDir)
            addpath(utilsDir);
        end
    catch
        % Path setup is optional; proceed if it fails.
    end

    % ---- sanity checks ----
    % Required inputs for the A-step pipeline.
    must_exist_file(args.TrainSplitPath, "TRAIN split file");
    must_exist_file(args.ValidSplitPath, "VALID split file");
    must_exist_file(args.CalibPath,    "OLS calibration file");
    must_exist_file(args.WeightsPath,    "measurement weights file");

    % Optional (archive-only). Downstream logic does not require it.
    if ~isfile(args.TrainParticipantsProbesPath)
        warning("[A1] TrainParticipantsProbesPath missing (archive-only): %s", string(args.TrainParticipantsProbesPath));
    end

    % ---- run directory ----
    % Create a run-local directory structure to avoid overwriting shared derived/ outputs.
    outDir = string(args.OutDir);
    baseRunDir = fullfile(outDir, run_id);
    stepDir = fullfile(baseRunDir, "stepA1_prepare_analysis");
    
    ensure_dir(outDir);
    ensure_dir(baseRunDir);
    ensure_dir(stepDir);

    % ---- default run-local outputs ----
    % Default outputs are placed inside stepDir unless overridden by the caller.
    validMapped14Path = string(args.ValidMapped14Path);
    if strlength(validMapped14Path) == 0
        validMapped14Path = fullfile(stepDir, "participants_valid_mapped14_stepM2.mat");
    end
    
    validProbesMappedPath = string(args.ValidProbesMappedPath);
    if strlength(validProbesMappedPath) == 0
        validProbesMappedPath = fullfile(stepDir, "participants_valid_probes_mapped_stepM4.mat");
    end

    % Overwrite protection
    % Refuse to overwrite run outputs unless explicitly allowed.
    if ~args.Overwrite
        if isfile(validMapped14Path)
            error("[A1] Output exists (set Overwrite=true to replace): %s", validMapped14Path);
        end
        if isfile(validProbesMappedPath)
            error("[A1] Output exists (set Overwrite=true to replace): %s", validProbesMappedPath);
        end
    end

    % ---- load TRAIN ids for provenance checks ----
    % TRAIN participant IDs are used to verify that calibration artifacts were computed
    % on the intended training set.
    S_train = load(string(args.TrainSplitPath), "participants_clean", "info");
    trainP = S_train.participants_clean;
    trainIDs = read_participant_ids(trainP);

    % ---- load calibration ids ----
    % calib: expected to include calibration parameters and (optionally) the IDs used.
    S_c = load(string(args.CalibPath), "calib");
    calib = S_c.calib;


    % ---- provenance checks ----
    % Compare ID sets between calibration artifacts and TRAIN IDs, if the calibration
    % structs expose an "ids" field. Mismatches are warnings by default (non-fatal).
    prov = struct();
    if isfield(calib, "ids")
        prov.calib_vs_train = compare_id_sets(calib.ids, trainIDs, "calib.ids", "train.participant_id");
    else
        prov.calib_vs_train = struct("ok", false, "summary", "calib.ids not present; cannot verify provenance.");
    end

    if isfield(prov, "calib_vs_train") && ~prov.calib_vs_train.ok
        warning("[A1] Provenance mismatch: %s", prov.calib_vs_train.summary);
    end

    % ---- apply TRAIN calibration to VALID ----
    % Apply mapping models derived from TRAIN to the VALID split:
    %   - Step M2: map 14-item questionnaire to the 40-item trust scale
    %   - Step M4: map probe (single-item) trust to the 40-item trust scale
    fprintf("[A1] Applying training calibration to validation set...\n");
    fprintf("     VALID input: %s\n", string(args.ValidSplitPath));
    fprintf("     calib:     %s\n", string(args.CalibPath));
    fprintf("     VALID out14:  %s\n", validMapped14Path);
    fprintf("     VALID outP:   %s\n", validProbesMappedPath);

    % M2 (valid): uses training calib14
    stepM2_apply_14mapping(string(args.ValidSplitPath), ...
                           string(args.CalibPath), ...
                           string(validMapped14Path));

    % M4 (valid): uses training calibProbe; input is VALID mapped14 from M2
    stepM4_apply_probe_mapping(string(validMapped14Path), ...
                               string(args.CalibPath), ...
                               string(validProbesMappedPath));

    % ---- archive/copy key artifacts into stepDir ----
    % Archive all inputs and key outputs into the run-local directory for reproducibility.
    fprintf("[A1] Archiving artifacts into: %s\n", stepDir);

    % Split definition files
    copy_into_dir(stepDir, args.TrainSplitPath, "Overwrite", true);
    copy_into_dir(stepDir, args.ValidSplitPath, "Overwrite", true);

    % Calibration + weights
    copy_into_dir(stepDir, args.CalibPath,    "Overwrite", true);
    copy_into_dir(stepDir, args.WeightsPath,    "Overwrite", true);

    % Train mapped probes (archive-only; optional)
    safe_copy_into_dir(stepDir, args.TrainParticipantsProbesPath, "Overwrite", true);

    % VALID mapped outputs (copy if caller provided custom output paths)
    if ~strcmp(char(validMapped14Path), fullfile(stepDir, "participants_valid_mapped14_stepM2.mat"))
        copy_into_dir(stepDir, validMapped14Path, "Overwrite", true);
    end
    if ~strcmp(char(validProbesMappedPath), fullfile(stepDir, "participants_valid_probes_mapped_stepM4.mat"))
        copy_into_dir(stepDir, validProbesMappedPath, "Overwrite", true);
    end

    % ---- optionally write canonical convenience copies into derived/ ----
    % This is intended for workflows that expect fixed filenames in derived/.
    if args.WriteCanonicalLatest
        fprintf("[A1] Writing canonical latest copies into derived/ (overwriting)...\n");
        copyfile(validMapped14Path,  "derived/participants_valid_mapped14_stepM2.mat");
        copyfile(validProbesMappedPath, "derived/participants_valid_probes_mapped_stepM4.mat");
    end

    % ---- save manifest ----
    % The manifest captures split IDs, file metadata, provenance checks, and key scalar
    % calibration/weight values commonly referenced by downstream steps and reports.
    runInfo = struct();
    runInfo.run_id  = char(run_id);
    runInfo.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    runInfo.archive_dir = char(stepDir);
    runInfo.base_run_dir = char(baseRunDir);

    % File metadata
    runInfo.files = struct();
    runInfo.files.train_split                 = file_info_struct(args.TrainSplitPath);
    runInfo.files.valid_split                 = file_info_struct(args.ValidSplitPath);
    runInfo.files.calib                       = file_info_struct(args.CalibPath);
    runInfo.files.weights                     = file_info_struct(args.WeightsPath);
    runInfo.files.train_mapped_probes_archive = file_info_struct(args.TrainParticipantsProbesPath);
    runInfo.files.valid_mapped14_out          = file_info_struct(validMapped14Path);
    runInfo.files.valid_mappedP_out           = file_info_struct(validProbesMappedPath);

    % Split info (IDs + counts)
    runInfo.split = struct();
    runInfo.split.n_train = numel(trainP);
    runInfo.split.train_ids = cellstr(trainIDs);

    S_valid = load(string(args.ValidSplitPath), "participants_clean", "info");
    validP = S_valid.participants_clean;
    validIDs = read_participant_ids(validP);

    runInfo.split.n_valid = numel(validP);
    runInfo.split.valid_ids = cellstr(validIDs);

    % Include split metadata structs if present
    if isfield(S_train, "info"), runInfo.split.train_info = S_train.info; end
    if isfield(S_valid, "info"), runInfo.split.valid_info = S_valid.info; end

    % Provenance checks
    runInfo.provenance = prov;

    % Calibration summary (store scalar parameters if present)
    runInfo.calibration = struct();
    if isfield(calib, "a14"), runInfo.calibration.a14 = calib.a14; end
    if isfield(calib, "b14"), runInfo.calibration.b14 = calib.b14; end
    if isfield(calib, "a1"), runInfo.calibration.a1 = calib.a1; end
    if isfield(calib, "b1"), runInfo.calibration.b1 = calib.b1; end

    % Weights summary
    S_w = load(string(args.WeightsPath), "weights");
    W = S_w.weights;
    runInfo.weights = struct("w40", W.w40, "w14", W.w14, "w_probe", W.w_probe);

    save(fullfile(stepDir, "manifest.mat"), "runInfo", "-v7.3");
    save_json(fullfile(stepDir, "manifest.json"), runInfo);

    fprintf("[A1] Done.\n");
    fprintf("     Manifest saved to %s\n", fullfile(stepDir, "manifest.mat"));
end
