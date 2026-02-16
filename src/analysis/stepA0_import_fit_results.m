function [run_id, resultsMatPath, runInfo] = stepA0_import_fit_results(cfg, varargin)
% stepA0_import_fit_results  Import and repackage trust-fit results for A-step analysis.
%
% This step provides a minimal interface layer between the trust optimisation
% pipeline and the Step A analysis pipeline (A1–A6).
%
% It:
%   1) Infers the fit run directory from cfg.run_tag and cfg.dt (or uses an override),
%   2) Loads the final optimisation summary and manifest produced by
%      run_trust_optimisation_pipeline,
%   3) Repackages fitted parameter vectors into an A2-compatible MAT file
%      exposing theta_hat_* fields and cfg.dt,
%   4) Writes run-local outputs and a small manifest for provenance/auditability.
%
% No model behavior is changed. No refitting or recomputation is performed.
%
% Inputs
%   cfg (struct)
%       Configuration struct used for the trust optimisation pipeline.
%       Required fields:
%         - cfg.dt       (numeric scalar)
%         - cfg.run_tag  (string/char)
%
% Name-Value Arguments (all optional)
%   "FitBaseDir" (string/char)
%       Base directory containing fit runs.
%       Default: "derived/fit_runs"
%
%   "FitRunDir" (string/char)
%       Explicit path to a fit run directory. If provided, overrides FitBaseDir
%       and inferred run_id.
%       Default: ""
%
%   "OutDir" (string/char)
%       Base output directory for analysis runs.
%       Default: "derived/analysis_runs"
%
%   "ResultsMatName" (string/char)
%       Filename for the repackaged results MAT file.
%       Default: "fit_results_stepA0.mat"
%
%   "Overwrite" (logical scalar)
%       If false, refuses to overwrite existing outputs.
%       Default: false
%
% Outputs
%   run_id (string)
%       Run identifier derived from cfg.run_tag and cfg.dt. Intended to be
%       passed directly to Step A1 and subsequent A-steps.
%
%   resultsMatPath (string)
%       Full path to the A2-compatible results MAT file written by this step.
%
%   runInfo (struct)
%       Manifest struct summarizing source files, mappings, and outputs.
%       Also saved to <stepDir>/manifest.mat.
%
% Assumptions / Dependencies
%   - Fit pipeline has completed successfully and produced:
%       <FitRunDir>/manifest.mat
%       <FitRunDir>/final/run_summary.mat
%   - No utilities are strictly required, but if file_info_struct or save_json
%     exist on the path, they may be used in future extensions.

    % ---------------------------------------------------------------------
    % 0) Input validation
    % ---------------------------------------------------------------------
    if nargin < 1 || ~isstruct(cfg)
        error("stepA0_import_fit_results: cfg struct is required.");
    end

    if ~isfield(cfg, "dt") || isempty(cfg.dt)
        error("cfg.dt is required.");
    end
    if ~isfield(cfg, "run_tag") || isempty(cfg.run_tag)
        error("cfg.run_tag is required.");
    end

    if ~isscalar(cfg.dt) || ~isnumeric(cfg.dt)
        error("cfg.dt must be a numeric scalar.");
    end

    % ---------------------------------------------------------------------
    % 1) Parse name-value arguments
    % ---------------------------------------------------------------------
    p = inputParser;
    p.addParameter("FitBaseDir", "derived/fit_runs", @(s) isstring(s) || ischar(s));
    p.addParameter("FitRunDir",  "", @(s) isstring(s) || ischar(s));
    p.addParameter("OutDir",     "derived/analysis_runs", @(s) isstring(s) || ischar(s));
    p.addParameter("ResultsMatName", "fit_results_stepA0.mat", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));

    p.parse(varargin{:});
    args = p.Results;

    % ---------------------------------------------------------------------
    % 2) Derive run_id (must match fit pipeline logic)
    % ---------------------------------------------------------------------
    dt_str = regexprep(sprintf("dt%.3g", cfg.dt), '\.', 'p');
    run_id = string(cfg.run_tag) + "__" + dt_str;

    % ---------------------------------------------------------------------
    % 3) Resolve fit run directory
    % ---------------------------------------------------------------------
    if strlength(string(args.FitRunDir)) > 0
        fitRunDir = string(args.FitRunDir);
    else
        fitRunDir = fullfile(string(args.FitBaseDir), run_id);
    end

    if ~isfolder(fitRunDir)
        error("Fit run directory not found: %s", fitRunDir);
    end

    fitManifestPath = fullfile(fitRunDir, "manifest.mat");
    fitSummaryPath  = fullfile(fitRunDir, "final", "run_summary.mat");

    if ~isfile(fitManifestPath)
        error("Fit manifest not found: %s", fitManifestPath);
    end
    if ~isfile(fitSummaryPath)
        error("Fit run summary not found: %s", fitSummaryPath);
    end

    % ---------------------------------------------------------------------
    % 4) Load and validate fit manifest
    % ---------------------------------------------------------------------
    S_m = load(fitManifestPath, "manifest");
    if ~isfield(S_m, "manifest")
        error("Fit manifest.mat does not contain variable 'manifest'.");
    end
    fitManifest = S_m.manifest;

    if ~isfield(fitManifest, "run_id") || string(fitManifest.run_id) ~= run_id
        error("Fit manifest run_id mismatch (expected %s).", run_id);
    end

    if ~isfield(fitManifest, "cfg") || ~isfield(fitManifest.cfg, "dt")
        error("Fit manifest missing cfg.dt.");
    end

    if abs(double(fitManifest.cfg.dt) - double(cfg.dt)) > 1e-12
        error("cfg.dt does not match dt used in fit run.");
    end

    % ---------------------------------------------------------------------
    % 5) Load fit summary (stage results)
    % ---------------------------------------------------------------------
    S_s = load(fitSummaryPath, "summary");
    if ~isfield(S_s, "summary")
        error("run_summary.mat does not contain variable 'summary'.");
    end
    summary = S_s.summary;

    if ~isfield(summary, "stage_results") || isempty(summary.stage_results)
        error("Fit summary does not contain stage_results.");
    end

    % ---------------------------------------------------------------------
    % 6) Prepare output directories
    % ---------------------------------------------------------------------
    outBaseDir = string(args.OutDir);
    baseRunDir = fullfile(outBaseDir, run_id);
    stepDir    = fullfile(baseRunDir, "stepA0_import_fit");

    if ~isfolder(outBaseDir), mkdir(outBaseDir); end
    if ~isfolder(baseRunDir), mkdir(baseRunDir); end
    if ~isfolder(stepDir), mkdir(stepDir); end

    resultsMatPath = fullfile(stepDir, string(args.ResultsMatName));

    if ~args.Overwrite && isfile(resultsMatPath)
        error("Output exists (set Overwrite=true to replace): %s", resultsMatPath);
    end

    % ---------------------------------------------------------------------
    % 7) Repackage fitted parameters
    % ---------------------------------------------------------------------
    R = struct();

    % Minimal cfg required by A2
    R.cfg = struct();
    R.cfg.dt = double(cfg.dt);

    mappings = struct([]);
    map_idx = 0;

    for k = 1:numel(summary.stage_results)
        sr = summary.stage_results(k);

        if ~isfield(sr, "theta_hat") || isempty(sr.theta_hat)
            continue;
        end

        method = string(sr.method);
        preset = string(sr.preset);

        % Build a deterministic, valid field name
        fname = sprintf("theta_hat_stage%02d_%s", k, method);
        if strlength(preset) > 0
            fname = fname + "__" + preset;
        end
        fname = matlab.lang.makeValidName(fname);

        R.(fname) = sr.theta_hat(:);  % preserve values, normalize shape only

        map_idx = map_idx + 1;
        mappings(map_idx).stage = k;
        mappings(map_idx).method = char(method);
        mappings(map_idx).preset = char(preset);
        mappings(map_idx).output_field = char(fname);
    end

    % Best parameter vector (convenience)
    if isfield(summary, "best_theta") && ~isempty(summary.best_theta)
        R.theta_hat_best = summary.best_theta(:);
    end

    % Optionally include the full summary for traceability
    R.fit_summary = summary;

    save(resultsMatPath, "-struct", "R", "-v7.3");

    % ---------------------------------------------------------------------
    % 8) Write Step A0 manifest
    % ---------------------------------------------------------------------
    runInfo = struct();
    runInfo.run_id   = char(run_id);
    runInfo.created  = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));

    runInfo.source = struct();
    runInfo.source.fit_run_dir      = char(fitRunDir);
    runInfo.source.fit_manifest_mat = char(fitManifestPath);
    runInfo.source.fit_summary_mat  = char(fitSummaryPath);

    runInfo.mapping = mappings;

    runInfo.outputs = struct();
    runInfo.outputs.results_mat = char(resultsMatPath);

    runInfo.determinism = ...
        "Step A0 performs a deterministic, value-preserving repackaging of fit outputs. No refitting or recomputation is performed.";

    save(fullfile(stepDir, "manifest.mat"), "runInfo", "-v7.3");

    fprintf("[A0] Imported fit results for run_id: %s\n", run_id);
    fprintf("     Fit source: %s\n", fitRunDir);
    fprintf("     Output:     %s\n", resultsMatPath);
end
