function results = run_trust_optimisation_pipeline(cfg)
% run_trust_optimisation_pipeline  Orchestrate trust optimisation pipeline
%
% Executes a fixed 4-stage optimisation pipeline:
%   1) GA (overnight preset) – global exploration
%   2) GA (overnight preset) – stability check
%   3) fmincon        – local refinement (seeded from best GA)
%   4) patternsearch (overnight preset) – robustness validation
%
% The pipeline is crash-safe and restartable via checkpoints.
% After successful completion, final artefacts are written and
% checkpoints are deleted. Rerunning the pipeline archives any
% previous run folder automatically.
%
% Required cfg fields:
%   cfg.dt      - simulation timestep (seconds)
%   cfg.theta0  - 8x1 initial parameter vector
%
% Optional cfg fields:
%   cfg.run_tag     - descriptive tag (default: "trust_fit")
%   cfg.base_dir    - base results directory (default: "derived/fit_runs")

    cfg = apply_defaults(cfg);

    % -------------------------
    % Run identity & directories
    % -------------------------
    run_id = cfg.run_id;
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

    fprintf('[pipeline] Run ID: %s\n', run_id);
    fprintf('[pipeline] Started: %s\n', datestr(now));
    fprintf('[pipeline] dt = %.3f\n', cfg.dt);

    % -------------------------
    % Manifest (written immediately)
    % -------------------------
    manifest = struct();
    manifest.run_id = run_id;
    manifest.created = datetime("now");
    manifest.cfg = cfg;
    manifest.stage_plan = stage_plan();
    manifest.completed_stages = false(4,1);
    save(fullfile(run_dir, "manifest.mat"), "manifest");

    % -------------------------
    % Load existing checkpoints if resuming
    % -------------------------
    stage_results = repmat(empty_stage_result(), 4, 1);
    for k = 1:4
        ckpt = fullfile(run_dir, "checkpoints", sprintf("stage_%02d.mat", k));
        if isfile(ckpt)
            load(ckpt, "stage_result");
            stage_results(k) = stage_result;
            manifest.completed_stages(k) = true;
        end
    end

    % -------------------------
    % Execute stages
    % -------------------------
    stages = stage_plan();

    for k = 1:4
        if manifest.completed_stages(k)
            fprintf('[pipeline] Stage %d already completed. Skipping.\n', k);
            continue;
        end

        fprintf('\n[pipeline] ==============================================\n');
        fprintf('[pipeline] Stage %d: %s (%s)\n', ...
            k, stages(k).method, stages(k).preset);
        fprintf('[pipeline] ==============================================\n');

        theta_init = select_theta_init(cfg, stage_results, k);

        t_start = tic;
        [theta_hat, fval, exitflag, output] = ...
            fit_trust_parameters(stages(k).method, cfg.dt, ...
                                 stages(k).preset, theta_init);
        runtime = toc(t_start);

        stage_result = empty_stage_result();
        stage_result.stage = k;
        stage_result.method = stages(k).method;
        stage_result.preset = stages(k).preset;
        stage_result.theta_init = theta_init;
        stage_result.theta_hat = theta_hat;
        stage_result.fval = fval;
        stage_result.exitflag = exitflag;
        stage_result.output = output;
        stage_result.runtime_s = runtime;
        stage_result.finished = datetime("now");

        save(fullfile(run_dir, "checkpoints", sprintf("stage_%02d.mat", k)), ...
             "stage_result", "-v7.3");

        stage_results(k) = stage_result;
        manifest.completed_stages(k) = true;
        save(fullfile(run_dir, "manifest.mat"), "manifest");

        fprintf('[pipeline] Stage %d complete (fval = %.6g, %.1f s)\n', ...
            k, fval, runtime);
    end

    % -------------------------
    % Finalization
    % -------------------------
    fprintf('\n[pipeline] All stages complete. Finalizing results.\n');

    for k = 1:4
        sr = stage_results(k);
        fname = sprintf("stage_%02d__%s", k, sr.method);
        if strlength(sr.preset) > 0
            fname = fname + "__" + sr.preset;
        end
        fname = fname + ".mat";

        save(fullfile(run_dir, "final", fname), "sr", "cfg", "-v7.3");
    end

    % Summary
    fvals = arrayfun(@(s) s.fval, stage_results);
    [best_fval, best_idx] = min(fvals);

    summary = struct();
    summary.run_id = run_id;
    summary.best_stage = best_idx;
    summary.best_theta = stage_results(best_idx).theta_hat;
    summary.best_fval = best_fval;
    summary.stage_results = stage_results;
    summary.total_runtime_s = sum([stage_results.runtime_s]);
    summary.completed = datetime("now");

    save(fullfile(run_dir, "final", "run_summary.mat"), "summary", "-v7.3");

    % -------------------------
    % Cleanup
    % -------------------------
    rmdir(fullfile(run_dir, "checkpoints"), 's');

    fprintf('[pipeline] Completed successfully.\n');
    diary off;

    results = summary;
end

% =====================================================================
% Helper functions
% =====================================================================

function cfg = apply_defaults(cfg)
    if ~isfield(cfg, "dt"), error("cfg.dt required"); end
    if ~isfield(cfg, "theta0"), error("cfg.theta0 required"); end
    if ~isfield(cfg, "run_tag"), cfg.run_tag = "trust_fit"; end
    if ~isfield(cfg, "base_dir"), cfg.base_dir = "derived/fit_runs"; end

    dt_str = regexprep(sprintf("dt%.3g", cfg.dt), '\.', 'p');
    cfg.run_id = string(cfg.run_tag) + "__" + dt_str;
end

function stages = stage_plan()
    stages = struct( ...
        "method", {"ga","ga","fmincon","patternsearch"}, ...
        "preset", {"overnight","overnight","","overnight"} );
end

function theta = select_theta_init(cfg, results, k)
    switch k
        case {1,2}
            theta = cfg.theta0;
        case 3
            [~, i] = min([results(1).fval, results(2).fval]);
            theta = results(i).theta_hat;
        case 4
            theta = results(3).theta_hat;
    end
end

function s = empty_stage_result()
    s = struct("stage",[], "method","", "preset","", ...
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
