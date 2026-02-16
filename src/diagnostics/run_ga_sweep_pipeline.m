function results = run_ga_sweep_pipeline(cfg, N)
% run_ga_sweep_pipeline  Run N GA runs (overnight preset) with checkpoints
%
% Runs N independent GA optimisations using:
%   method = "ga"
%   preset = "overnight"
%
% Crash-safe & restartable:
%   Each completed run is checkpointed as checkpoints/ga_run_XXX.mat
%   Rerunning resumes incomplete runs.
%
% Required:
%   cfg.dt   - simulation timestep (seconds)
%   N        - number of GA runs (positive integer)
%
% Optional:
%   cfg.run_tag  - descriptive tag (default: "trust_fit_ga_sweep")
%   cfg.base_dir - base results directory (default: "derived/fit_runs")

    % -------------------------
    % Validate + defaults
    % -------------------------
    if nargin < 2, error("N required"); end
    if ~isscalar(N) || N < 1 || fix(N) ~= N
        error("N must be a positive integer.");
    end
    cfg = apply_defaults(cfg);

    % -------------------------
    % Run identity & directories
    % -------------------------
    run_id  = cfg.run_id;
    base_dir = cfg.base_dir;
    run_dir = fullfile(base_dir, run_id);

    if isfolder(run_dir)
        archive_existing_run(base_dir, run_id);
    end

    mkdir(run_dir);
    mkdir(fullfile(run_dir, "logs"));
    mkdir(fullfile(run_dir, "checkpoints"));
    mkdir(fullfile(run_dir, "final"));

    % -------------------------
    % Logging
    % -------------------------
    diary_file = fullfile(run_dir, "logs", "pipeline.log");
    diary off;
    diary(diary_file);

    fprintf('[ga_sweep] Run ID: %s\n', run_id);
    fprintf('[ga_sweep] Started: %s\n', datestr(now));
    fprintf('[ga_sweep] dt = %.3f\n', cfg.dt);
    fprintf('[ga_sweep] N  = %d\n', N);

    % -------------------------
    % Manifest (written immediately)
    % -------------------------
    manifest = struct();
    manifest.run_id = run_id;
    manifest.created = datetime("now");
    manifest.cfg = cfg;
    manifest.N = N;
    manifest.method = "ga";
    manifest.preset = "overnight";
    manifest.completed = false(N, 1);
    save(fullfile(run_dir, "manifest.mat"), "manifest");

    % -------------------------
    % Load checkpoints if resuming
    % -------------------------
    run_results = repmat(empty_run_result(), N, 1);
    for i = 1:N
        ckpt = fullfile(run_dir, "checkpoints", sprintf("ga_run_%03d.mat", i));
        if isfile(ckpt)
            load(ckpt, "run_result");
            run_results(i) = run_result;
            manifest.completed(i) = true;
        end
    end

    % Persist manifest updates after loading ckpts
    save(fullfile(run_dir, "manifest.mat"), "manifest");

    % -------------------------
    % Execute GA runs
    % -------------------------
    method = "ga";
    preset = "overnight";
    theta_init = [];  % <-- no initial guess by design

    for i = 1:N
        if manifest.completed(i)
            fprintf('[ga_sweep] Run %d/%d already completed. Skipping.\n', i, N);
            continue;
        end

        fprintf('\n[ga_sweep] ==============================================\n');
        fprintf('[ga_sweep] GA run %d/%d (%s preset)\n', i, N, preset);
        fprintf('[ga_sweep] ==============================================\n');

        t_start = tic;
        [theta_hat, fval, exitflag, output] = ...
            fit_trust_parameters(method, cfg.dt, preset, theta_init);
        runtime = toc(t_start);

        run_result = empty_run_result();
        run_result.index = i;
        run_result.method = method;
        run_result.preset = preset;
        run_result.theta_init = theta_init;
        run_result.theta_hat = theta_hat;
        run_result.fval = fval;
        run_result.exitflag = exitflag;
        run_result.output = output;
        run_result.runtime_s = runtime;
        run_result.finished = datetime("now");

        save(fullfile(run_dir, "checkpoints", sprintf("ga_run_%03d.mat", i)), ...
             "run_result", "-v7.3");

        run_results(i) = run_result;
        manifest.completed(i) = true;
        save(fullfile(run_dir, "manifest.mat"), "manifest");

        fprintf('[ga_sweep] Run %d complete (fval = %.6g, %.1f s)\n', ...
            i, fval, runtime);
    end

    % -------------------------
    % Finalization
    % -------------------------
    fprintf('\n[ga_sweep] All GA runs complete. Finalizing results.\n');

    % Save all runs together
    save(fullfile(run_dir, "final", "ga_sweep_all_runs.mat"), ...
         "run_results", "cfg", "N", "-v7.3");

    % Find best
    fvals = arrayfun(@(r) r.fval, run_results);
    [best_fval, best_idx] = min(fvals);

    summary = struct();
    summary.run_id = run_id;
    summary.N = N;
    summary.best_run = best_idx;
    summary.best_theta = run_results(best_idx).theta_hat;
    summary.best_fval = best_fval;
    summary.run_results = run_results;
    summary.total_runtime_s = sum([run_results.runtime_s]);
    summary.completed = datetime("now");

    save(fullfile(run_dir, "final", "run_summary.mat"), "summary", "-v7.3");

    % Cleanup checkpoints (optional; mirrors your original behavior)
    rmdir(fullfile(run_dir, "checkpoints"), 's');

    fprintf('[ga_sweep] Completed successfully.\n');
    diary off;

    results = summary;
end

% =====================================================================
% Helper functions
% =====================================================================

function cfg = apply_defaults(cfg)
    if ~isfield(cfg, "dt"), error("cfg.dt required"); end
    if ~isfield(cfg, "run_tag"), cfg.run_tag = "trust_fit_ga_sweep"; end
    if ~isfield(cfg, "base_dir"), cfg.base_dir = "derived/fit_runs"; end

    dt_str = regexprep(sprintf("dt%.3g", cfg.dt), '\.', 'p');
    cfg.run_id = string(cfg.run_tag) + "__" + dt_str;
end

function r = empty_run_result()
    r = struct("index",[], "method","", "preset","", ...
               "theta_init",[], "theta_hat",[], ...
               "fval",NaN, "exitflag",NaN, ...
               "output",struct(), "runtime_s",NaN, ...
               "finished",datetime.empty);
end

function archive_existing_run(base_dir, run_id)
    old_dir = fullfile(base_dir, "old");
    if ~isfolder(old_dir), mkdir(old_dir); end
    ts = datestr(now, "yyyymmdd_HHMMSS");
    movefile(fullfile(base_dir, run_id), ...
             fullfile(old_dir, run_id + "__archived__" + ts));
end
