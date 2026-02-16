function stepA6_report_baseline_comparison_simple_mode(run_id, inA5Dir)
% stepA6_report_baseline_comparison_simple_mode Generate SIMPLE-mode baseline-comparison report from Step A5 outputs.
%
% This step creates a thesis-ready reporting bundle for the SIMPLE-mode baseline
% comparison using the consolidated MAT output from Step A5 (A5_plot_data.mat).
% It intentionally avoids recomputing metrics from raw CSV sources to ensure
% consistent reporting across the analysis pipeline.
%
% In addition to aggregate tables and figures, this step generates per-participant
% validation trajectory plots by re-simulating:
%   - the fitted SIMPLE model via trust_simulate_or_predict_one_participant("simple", ...)
%   - each baseline via trust_simulate_baseline_one_participant(method, ...)
%
% INPUTS
%   run_id (string|char)
%       Analysis run identifier. Outputs are written under:
%         derived/analysis_runs/<run_id>/stepA6_report/
%
%   inA5Dir (string|char) (optional)
%       Directory containing Step A5 outputs. If omitted/empty, defaults to:
%         derived/analysis_runs/<run_id>/stepA5_baseline_comparison
%
% OUTPUTS
%   (none)
%       Writes CSV tables, text summary, and figures into:
%         derived/analysis_runs/<run_id>/stepA6_report/
%       and per-participant trajectory figures into:
%         derived/analysis_runs/<run_id>/stepA6_report/participant_trajectories/<participant_id>/
%
% ASSUMPTIONS / DEPENDENCIES
%   - Step A5 produced: <inA5Dir>/A5_plot_data.mat containing variable A5plot
%     with fields: methods, validOverall, validKind, improvements, bestBaseline_valid,
%     bestBaseline_valid_wRMSE, dt, fit_params, (optional) meta.theta_star and cfg.
%   - Step A1 archived validation participants and weights under:
%       derived/analysis_runs/<run_id>/stepA1_prepare_analysis/
%         participants_valid_probes_mapped_stepM4.mat
%         measurement_weights.mat
%   - Utilities available on the MATLAB path:
%       ensure_dir, load_participants_struct, read_participant_ids
%   - Simulation helpers available on the MATLAB path:
%       trust_simulate_or_predict_one_participant, trust_simulate_baseline_one_participant
%   - Plot helpers available on the MATLAB path:
%       thesisStyle, thesisFinalizeFigure, thesisExport, methodDisplayName,
%       measurementDisplayName
%
% NOTES
%   - Baseline method IDs (e.g., "bump_asymmetric") are preserved for compatibility
%     with Step A5 artifacts and the baseline simulator dispatch.
%   - Trajectory plots overlay door-event markers and aligned measurement markers.

    % -------------------------
    % Validate/normalize inputs
    % -------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA6_report_baseline_comparison_simple_mode: run_id is required.");
    end
    run_id = string(run_id);

    if nargin < 2 || isempty(inA5Dir)
        inA5Dir = fullfile("derived","analysis_runs",run_id,"stepA5_baseline_comparison");
    end
    inA5Dir = string(inA5Dir);

    if ~isfolder(inA5Dir)
        error("A5 folder not found: %s", inA5Dir);
    end

    % ------------------------------------------------------------------
    % Load consolidated A5 plot bundle (single source of truth for tables)
    % ------------------------------------------------------------------
    plotMat = fullfile(inA5Dir, "A5_plot_data.mat");
    if ~isfile(plotMat)
        error("A5_plot_data.mat not found. Re-run updated Step A5. Missing: %s", plotMat);
    end

    S = load(plotMat, "A5plot");
    if ~isfield(S, "A5plot")
        error("A5_plot_data.mat does not contain variable 'A5plot'.");
    end
    A5plot = S.A5plot;

    % ------------------------------------------------------------------
    % Output directory
    % ------------------------------------------------------------------
    outDir = fullfile("derived","analysis_runs",run_id,"stepA6_report");
    if exist("ensure_dir","file") == 2
        ensure_dir(outDir);
    else
        if ~isfolder(outDir), mkdir(outDir); end
    end

    % ------------------------------------------------------------------
    % Pull core reporting tables/bundles from A5 plot data
    % ------------------------------------------------------------------
    methods      = string(A5plot.methods);
    validOverall = A5plot.validOverall;
    validKind    = A5plot.validKind;
    improvements = A5plot.improvements;

    bestBaseline = string(A5plot.bestBaseline_valid);
    bestVal      = double(A5plot.bestBaseline_valid_wRMSE);

    % Basic consistency check: bestBaseline should be listed in A5 methods
    if ~ismember(bestBaseline, methods) && ~(ismissing(bestBaseline) || strlength(bestBaseline)==0)
        warning("[A6] bestBaseline '%s' not found in A5plot.methods.", bestBaseline);
    end

    % Reorder validOverall rows to match the methods ordering (if possible)
    if istable(validOverall) && ismember("method", validOverall.Properties.VariableNames)
        vMethods = string(validOverall.method);
        if ~isequal(vMethods(:), methods(:))
            [tf, loc] = ismember(methods, vMethods);
            if all(tf)
                validOverall = validOverall(loc, :);
            else
                warning("[A6] Could not fully reorder validOverall to match methods.");
            end
        end
    end

    % ------------------------------------------------------------------
    % Export reporting tables as CSV (no metric recomputation in A6)
    % ------------------------------------------------------------------
    writetable(validOverall, fullfile(outDir, "A6_valid_overall_table.csv"));
    writetable(validKind,    fullfile(outDir, "A6_valid_kind_table.csv"));

    if ~isempty(improvements) && istable(improvements) && height(improvements) > 0
        impValid = improvements(string(improvements.split) == "valid", :);
        writetable(impValid, fullfile(outDir, "A6_valid_improvements_table.csv"));
    else
        impValid = table(); %#ok<NASGU>
        warning("[A6] No improvements table available in A5plot bundle.");
    end

    % Save best baseline selection (human-readable)
    fid = fopen(fullfile(outDir, "A6_best_baseline.txt"), "w");
    fprintf(fid, "Best baseline on VALID (overall wRMSE): %s (wRMSE=%.6g)\n", bestBaseline, bestVal);
    fclose(fid);

    % ------------------------------------------------------------------
    % Meta information (provenance for reporting)
    % ------------------------------------------------------------------
    meta = struct();
    meta.run_id = char(run_id);
    meta.inA5Dir = char(inA5Dir);
    meta.outDir = char(outDir);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.bestBaseline_valid = char(bestBaseline);
    meta.bestBaseline_valid_wRMSE = bestVal;
    meta.source_plot_bundle = char(plotMat);
    meta.note = "A6 uses A5_plot_data.mat for tables; trajectories re-load VALID participants from A1 archive and simulate via trust_simulate_* functions.";
    save(fullfile(outDir, "meta.mat"), "meta");

    % Load plotting defaults/style (project-specific helper)
    Sth = thesisStyle();

    % ------------------------------------------------------------------
    % FIG 1: Validation overall wRMSE by method
    % ------------------------------------------------------------------
    fig1 = figure('Visible','off','Color','w','Name','VALID overall wRMSE');
    thesisStyle(fig1);

    vals = validOverall.wRMSE_overall;

    % Internal method IDs must be valid categorical category keys.
    methods = string(methods);
    methods = strip(methods);

    % Defensive fallback to avoid invalid categorical category names
    badID = ismissing(methods) | strlength(methods)==0;
    if any(badID)
        methods(badID) = "UNKNOWN";
    end

    % Display labels used for axis tick labels
    methodLabels = arrayfun(@methodDisplayName, methods);
    methodLabels = strip(methodLabels);

    badLbl = ismissing(methodLabels) | strlength(methodLabels)==0;
    if any(badLbl)
        methodLabels(badLbl) = methods(badLbl);
    end

    cats = categorical(string(validOverall.method), cellstr(methods), cellstr(methodLabels));
    bar(cats, vals);

    grid on;
    ylabel("wRMSE");
    title("Validation overall wRMSE by method");
    ax = gca;
    ax.XTickLabelRotation = 25;

    thesisFinalizeFigure(fig1, Sth);
    thesisExport(fig1, fullfile(outDir, "fig_valid_overall_wrmse"));

    % ------------------------------------------------------------------
    % FIG 2: Validation wRMSE by measurement kind (grouped bars)
    % ------------------------------------------------------------------
    if ~istable(validKind) || height(validKind) == 0
        warning("[A6] validKind table missing/empty; skipping fig_valid_wrmse_by_kind.png");
    else
        % Ensure there is a single row per (method, kind)
        mk = string(validKind.method) + "||" + string(validKind.kind);
        [u, ~, ic] = unique(mk);
        counts = accumarray(ic, 1);
        if any(counts > 1)
            bad = u(counts > 1);
            error("[A6] validKind has duplicate (method,kind) rows. Examples:\n  %s", strjoin(bad(1:min(10,numel(bad))), "\n  "));
        end

        kindsAll = sort(unique(string(validKind.kind)));
        M = numel(methods);
        K = numel(kindsAll);
        matWR = NaN(M, K);

        % Assemble MxK matrix of wRMSE values (NaN if missing)
        for i = 1:M
            m = methods(i);
            for k = 1:K
                kk = kindsAll(k);
                row = validKind(string(validKind.method) == m & string(validKind.kind) == kk, :);
                if ~isempty(row)
                    matWR(i,k) = row.wRMSE(1);
                end
            end
        end

        fig2 = figure('Visible','off','Color','w','Name','VALID wRMSE by kind');
        thesisStyle(fig2);

        bar(matWR, 'grouped');
        grid on;

        % Explicit color order for deterministic appearance across runs
        set(gca, 'ColorOrder', [Sth.colors.cyan; Sth.colors.yellow; Sth.colors.green; Sth.colors.burgundy], 'NextPlot', 'replacechildren');

        xlabel("model");
        ylabel("wRMSE");
        title("Validation wRMSE by measurement type");

        ax = gca;
        ax.XTick = 1:M;
        methodLabelsCell = arrayfun(@methodDisplayName, methods, 'UniformOutput', false);
        ax.XTickLabel = methodLabelsCell;
        ax.XTickLabelRotation = 25;

        kindslabelsCell = arrayfun(@measurementDisplayName, kindsAll, 'UniformOutput', false);
        legend(cellstr(kindslabelsCell), 'Location','bestoutside');

        thesisFinalizeFigure(fig2, Sth);
        thesisExport(fig2, fullfile(outDir, "fig_valid_wrmse_by_kind"));
    end

    % ------------------------------------------------------------------
    % FIG 3: Participant-level wRMSE distribution (model vs best baseline)
    % ------------------------------------------------------------------
    Tjoin = table();
    canFig3 = true;

    if ~isfield(A5plot, "allParticipants") || ~isstruct(A5plot.allParticipants), canFig3 = false; end
    if canFig3 && ~isfield(A5plot.allParticipants, "simple_selected"), canFig3 = false; end
    if canFig3 && ~isfield(A5plot.allParticipants.simple_selected, "valid"), canFig3 = false; end
    if canFig3 && ~(isfield(A5plot.allParticipants, bestBaseline) && isfield(A5plot.allParticipants.(bestBaseline), "valid"))
        canFig3 = false;
    end

    if ~canFig3 || ismissing(bestBaseline) || strlength(bestBaseline)==0
        warning("[A6] Missing participant tables for model/bestBaseline; skipping fig_participant_wrmse_model_vs_bestbaseline.png");
    else
        Tp_model = A5plot.allParticipants.simple_selected.valid;
        Tp_base  = A5plot.allParticipants.(bestBaseline).valid;

        if ~istable(Tp_model) || ~istable(Tp_base)
            warning("[A6] Participant tables are not tables; skipping fig 3.");
        else
            mustCols = ["participant_id","wRMSE","N"];
            for c = mustCols
                if ~ismember(c, string(Tp_model.Properties.VariableNames))
                    error("Tp_model missing required column '%s'.", c);
                end
                if ~ismember(c, string(Tp_base.Properties.VariableNames))
                    error("Tp_base missing required column '%s'.", c);
                end
            end

            Tp_model.participant_id = string(Tp_model.participant_id);
            Tp_base.participant_id  = string(Tp_base.participant_id);

            Tp_model = Tp_model(:, {'participant_id','wRMSE','N'});
            Tp_base  = Tp_base(:,  {'participant_id','wRMSE','N'});

            % Rename columns for clarity after join
            Tp_model = local_rename_var(Tp_model, "wRMSE", "wRMSE_model");
            Tp_base  = local_rename_var(Tp_base,  "wRMSE", "wRMSE_baseline");
            Tp_model = local_rename_var(Tp_model, "N",     "N_model");
            Tp_base  = local_rename_var(Tp_base,  "N",     "N_baseline");

            % Join on participant_id (participants present in both tables only)
            Tjoin = innerjoin(Tp_model, Tp_base, 'Keys', 'participant_id');

            fig3 = figure('Visible','off','Color','w','Name','Participant wRMSE distribution');
            thesisStyle(fig3);

            data = [Tjoin.wRMSE_model, Tjoin.wRMSE_baseline];
            boxplot(data, 'Labels', { ...
                char(methodDisplayName("simple_selected")), ...
                char(methodDisplayName(bestBaseline)) ...
            });
            grid on;
            ylabel("participant-level wRMSE");
            title("Participant-level validation wRMSE: model vs best baseline");

            thesisFinalizeFigure(fig3, Sth);
            thesisExport(fig3, fullfile(outDir, "fig_participant_wrmse_model_vs_bestbaseline"));
        end
    end

    % ------------------------------------------------------------------
    % Participant trajectory plots (VALID split; all participants)
    %
    % For each participant, two figures are generated:
    %   (A) 3x2 tiled grid: up to five methods + a note panel with participant wRMSE.
    %   (B) Combo plot: constants (dashed) + model + best baseline + wRMSE note.
    % ------------------------------------------------------------------
    trajDir = fullfile(outDir, "participant_trajectories");
    if exist("ensure_dir","file") == 2
        ensure_dir(trajDir);
    else
        if ~isfolder(trajDir), mkdir(trajDir); end
    end

    a1Dir = fullfile("derived","analysis_runs",run_id,"stepA1_prepare_analysis");
    validMat = fullfile(a1Dir, "participants_valid_probes_mapped_stepM4.mat");
    weightsMat = fullfile(a1Dir, "measurement_weights.mat");

    if ~isfield(A5plot, "dt") || isempty(A5plot.dt)
        warning("[A6] A5plot.dt missing; skipping trajectory plots.");
    elseif ~isfield(A5plot, "fit_params") || ~isfield(A5plot.fit_params, "globalConstTrain")
        warning("[A6] A5plot.fit_params missing; skipping trajectory plots.");
    elseif ~isfile(validMat)
        warning("[A6] VALID participants file not found: %s. Skipping trajectory plots.", validMat);
    elseif ~isfile(weightsMat)
        warning("[A6] Weights file not found: %s. Skipping trajectory plots.", weightsMat);
    else
        participants_valid = load_participants_struct(validMat);

        W = load(weightsMat, "weights");
        if ~isfield(W, "weights")
            warning("[A6] weightsMat missing variable 'weights'; skipping trajectory plots.");
        else
            weights = W.weights;
            dt = double(A5plot.dt);

            % Parameter vector for the fitted SIMPLE model simulation
            theta_star = [];
            if isfield(A5plot, "meta") && isfield(A5plot.meta, "theta_star")
                theta_star = A5plot.meta.theta_star(:);
            elseif isfield(A5plot, "theta_star")
                theta_star = A5plot.theta_star(:);
            end
            if isempty(theta_star)
                warning("[A6] theta_star not found in A5plot bundle; skipping trajectory plots.");
            else
                % ----------------------------------------------------------
                % Configuration forwarded to baseline simulator (e.g., clipping)
                % ----------------------------------------------------------
                cfg = struct('clip01', true);

                if isfield(A5plot, "cfg") && isfield(A5plot.cfg, "clip01")
                    cfg.clip01 = logical(A5plot.cfg.clip01);
                elseif isfield(A5plot, "meta") && isfield(A5plot.meta, "cfg") && isfield(A5plot.meta.cfg, "clip01")
                    cfg.clip01 = logical(A5plot.meta.cfg.clip01);
                end

                if isfield(A5plot, "cfg") && isfield(A5plot.cfg, "optimo")
                    cfg.optimo = A5plot.cfg.optimo;
                elseif isfield(A5plot, "meta") && isfield(A5plot.meta, "cfg") && isfield(A5plot.meta.cfg, "optimo")
                    cfg.optimo = A5plot.meta.cfg.optimo;
                end

                % ----------------------------------------------------------
                % Baseline parameter bundle (passed through to baseline sim)
                % ----------------------------------------------------------
                fit_params = A5plot.fit_params;
                baselineParams = struct();
                baselineParams.globalConstTrain = double(fit_params.globalConstTrain);

                % Asymmetric bump baselines may require delta_plus/delta_minus
                dp = NaN; dm = NaN;
                if isfield(fit_params, "ode") && isstruct(fit_params.ode)
                    if isfield(fit_params.ode, "delta_plus"),  dp = double(fit_params.ode.delta_plus); end
                    if isfield(fit_params.ode, "delta_minus"), dm = double(fit_params.ode.delta_minus); end
                end
                if ~isfinite(dp), dp = 0; end
                if ~isfinite(dm), dm = 0; end
                baselineParams.delta_plus  = max(dp, 0);
                baselineParams.delta_minus = max(dm, 0);

                % Symmetric bump baseline uses a single delta parameter
                if isfield(fit_params, "bump") && isstruct(fit_params.bump) && isfield(fit_params.bump, "delta")
                    baselineParams.delta = max(double(fit_params.bump.delta), 0);
                else
                    baselineParams.delta = 0;
                end

                % Optimo baseline parameters (if present)
                if isfield(fit_params, "optimo") && isstruct(fit_params.optimo)
                    baselineParams.optimo = fit_params.optimo;
                end

                % Choose a baseline method for the combo plot (prefer A5 selection)
                bestBaselinePlot = bestBaseline;
                if ~ismember(bestBaselinePlot, methods)
                    bestBaselinePlot = local_choose_best_dynamic_baseline(validOverall, "bump_asymmetric", "bump_symmetric");
                end

                % Methods plotted in the 3x2 grid (fixed ordering)
                plotMethodsFixed = ["simple_selected", ...
                                    "optimo_lite", ...
                                    "bump_asymmetric", ...
                                    "optimo_lite_outcome_only", ...
                                    "bump_symmetric"];

                % Keep only those present in A5 outputs (always include simple_selected)
                plotMethods = strings(0,1);
                for i = 1:numel(plotMethodsFixed)
                    m = plotMethodsFixed(i);
                    if m == "simple_selected"
                        plotMethods(end+1) = m; %#ok<AGROW>
                    else
                        if ismember(m, methods)
                            plotMethods(end+1) = m; %#ok<AGROW>
                        end
                    end
                end

                % Map: key "method||participant_id" -> participant-level wRMSE (VALID)
                wrmseLookup = local_build_participant_wrmse_lookup(A5plot, methods);

                % Iterate all validation participants in the archived split
                pid_list = string(read_participant_ids(participants_valid));
                allRows = table();
                allRows.participant_id = pid_list(:);
                writetable(allRows, fullfile(trajDir, "all_participants_for_trajectories.csv"));

                for k = 1:numel(participants_valid)
                    P = participants_valid(k);
                    pid = pid_list(k);

                    % Participant-specific output folder
                    pidDir = fullfile(trajDir, char(pid));
                    if exist("ensure_dir","file") == 2
                        ensure_dir(pidDir);
                    else
                        if ~isfolder(pidDir), mkdir(pidDir); end
                    end

                    % Run model simulation once to obtain aligned measurements & door events
                    simModel = trust_simulate_or_predict_one_participant("simple", theta_star, P, dt);
                    [t_meas, y_meas, kind_meas] = local_unpack_measurements(simModel.measurements);

                    doorEvents = [];
                    if isfield(simModel, "doorEvents"), doorEvents = simModel.doorEvents; end

                    % (A) Grid plot: up to five methods + wRMSE note panel
                    local_plot_methods_grid(pidDir, pid, plotMethods, ...
                        P, dt, weights, cfg, theta_star, baselineParams, ...
                        t_meas, y_meas, kind_meas, doorEvents, wrmseLookup, Sth);

                    % (B) Combo plot: constants + model + best baseline
                    constMethods = ["const_dispositional","const_global_train_mean","const_oracle_participant_mean"];
                    local_plot_combo(pidDir, pid, bestBaselinePlot, constMethods, ...
                        P, dt, weights, cfg, theta_star, baselineParams, ...
                        t_meas, y_meas, kind_meas, doorEvents, wrmseLookup, Sth);
                end
            end
        end
    end

    % ------------------------------------------------------------------
    % Terminal summary
    % ------------------------------------------------------------------
    fprintf("\n[Step A6] VALID overall wRMSE (lower is better):\n");
    fprintf("  %-30s | %-12s\n", "method", "wRMSE");
    fprintf("  %s\n", repmat('-',1,48));
    for i = 1:height(validOverall)
        fprintf("  %-30s | %-12.6g\n", string(validOverall.method(i)), validOverall.wRMSE_overall(i));
    end
    fprintf("\n[Step A6] Best baseline on VALID: %s (wRMSE=%.6g)\n", bestBaseline, bestVal);
    fprintf("[Step A6] Output: %s\n", outDir);
    fprintf("[Step A6] Trajectories: %s\n", trajDir);
end

% =====================================================================
% Helpers: baseline selection and per-participant metric lookup
% =====================================================================

function best = local_choose_best_dynamic_baseline(validOverall, m1, m2)
% local_choose_best_dynamic_baseline Choose the better of two baseline methods on VALID.
%
% This helper selects the lower wRMSE_overall method between m1 and m2 using the
% validOverall summary table (if present and well-formed). If lookup fails, the
% function returns m1 as a conservative default.
%
% INPUTS
%   validOverall (table) - summary table containing method and wRMSE_overall
%   m1 (string|char)     - first baseline method ID
%   m2 (string|char)     - second baseline method ID
%
% OUTPUT
%   best (string) - chosen method ID

    best = m1;
    try
        if istable(validOverall) && ...
                ismember("method", validOverall.Properties.VariableNames) && ...
                ismember("wRMSE_overall", validOverall.Properties.VariableNames)
            T = validOverall;
            T.method = string(T.method);
            r1 = T.wRMSE_overall(T.method == m1);
            r2 = T.wRMSE_overall(T.method == m2);
            if ~isempty(r1) && ~isempty(r2) && isfinite(r2(1)) && isfinite(r1(1))
                if r2(1) < r1(1), best = m2; end
            elseif ~isempty(r2) && isfinite(r2(1))
                best = m2;
            end
        end
    catch
        best = m1;
    end
end

function mp = local_build_participant_wrmse_lookup(A5plot, methods)
% local_build_participant_wrmse_lookup Build (method,participant)->wRMSE lookup for VALID.
%
% Creates a containers.Map keyed by:
%   "method||participant_id"  (char)
% and storing:
%   participant-level wRMSE    (double)
%
% INPUTS
%   A5plot  (struct)          - loaded from A5_plot_data.mat
%   methods (string array)    - method IDs listed in A5plot.methods
%
% OUTPUT
%   mp (containers.Map)       - lookup map with KeyType char and ValueType double

    mp = containers.Map('KeyType','char','ValueType','double');
    if ~isfield(A5plot, "allParticipants") || ~isstruct(A5plot.allParticipants)
        return;
    end

    for i = 1:numel(methods)
        m = string(methods(i));
        if ~(isfield(A5plot.allParticipants, m) && isfield(A5plot.allParticipants.(m), "valid"))
            continue;
        end
        Tp = A5plot.allParticipants.(m).valid;
        if ~istable(Tp) || ~all(ismember(["participant_id","wRMSE"], string(Tp.Properties.VariableNames)))
            continue;
        end

        pid = string(Tp.participant_id);
        wr  = double(Tp.wRMSE);

        for k = 1:numel(pid)
            key = char(m + "||" + pid(k));
            if isfinite(wr(k))
                mp(key) = wr(k);
            end
        end
    end
end

function v = local_lookup_participant_wrmse(wrmseLookup, method, pid)
% local_lookup_participant_wrmse Retrieve participant-level wRMSE for a method (VALID).
%
% INPUTS
%   wrmseLookup (containers.Map) - from local_build_participant_wrmse_lookup()
%   method (string|char)         - method ID
%   pid (string|char)            - participant identifier
%
% OUTPUT
%   v (double) - participant wRMSE if present; NaN otherwise

    v = NaN;
    try
        key = char(string(method) + "||" + string(pid));
        if isKey(wrmseLookup, key)
            v = wrmseLookup(key);
        end
    catch
        v = NaN;
    end
end

% =====================================================================
% Plotting: participant trajectory figures
% =====================================================================

function local_plot_methods_grid(outDirPid, pid, plotMethods, P, dt, weights, cfg, theta_star, baselineParams, ...
    t_meas, y_meas, kind_meas, doorEvents, wrmseLookup, Sth)
% local_plot_methods_grid Multi-panel grid plot (3x2): up to five methods + a note panel.
%
% Panels 1..5: trust trajectories for the specified methods (one per tile).
% Panel 6: axis-off panel containing a note box with participant-level wRMSE values.
%
% The plot overlays:
%   - door-event times (vertical dotted lines)
%   - aligned trust measurements (marker-only by kind)

    fig = figure('Visible','off','Color','w','Name',sprintf('Traj %s methods_grid', pid));
    thesisStyle(fig);

    set(fig,'Units','centimeters');
    set(fig,'Position',[2 2 Sth.figSizeTrajectoryGrid]);

    plotMethods = string(plotMethods(:));
    if numel(plotMethods) > 5
        plotMethods = plotMethods(1:5);
    end

    % Plot horizon: stop shortly after the final measurement marker (if any)
    xPad = 30; % [s] extra time after last measurement

    % Create tiled layout (R2022b-safe)
    tl = tiledlayout(fig, 3, 2, 'TileSpacing','compact', 'Padding','compact');

    % Reserve top strips for title and legend (annotation + hidden axes)
    tl.Units = 'normalized';
    pos = tl.Position;

    titleStripH  = 0.000;
    legendStripH = 0.080;
    gap = 0.020;

    topReserve = titleStripH + gap + legendStripH + gap;
    pos(4) = max(0.10, pos(4) - topReserve);
    tl.Position = pos;

    % Title annotation (above legend strip)
    titleY = pos(2) + pos(4) + gap + legendStripH + gap;
    annotation(fig, 'textbox', [pos(1), titleY, pos(3), titleStripH], ...
        'String', sprintf('Participant %s - model vs. baselines', char(pid)), ...
        'Interpreter','none', ...
        'FontName', Sth.fontTitle, ...
        'FontSize', Sth.fontSizeTit, ...
        'FontWeight','normal', ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle',...
        'Tag','thesisTitle');

    % Legend host axes (hidden, used only to place a shared legend)
    legendY = pos(2) + pos(4) + gap;
    axLegendHost = axes(fig, 'Units','normalized', ...
        'Position',[pos(1), legendY, pos(3), legendStripH], ...
        'Visible','off');
    hold(axLegendHost,'on');

    % Capture legend sources from the first subplot
    hSimSrc  = gobjects(1,1);
    hMeasSrc = gobjects(0,1);

    % Determine x-axis maximum based on last measurement time (if available)
    tEndMeas = NaN;
    if ~isempty(t_meas)
        tEndMeas = max(double(t_meas(:)));
    end

    xMax = NaN;
    if isfinite(tEndMeas)
        xMax = tEndMeas + xPad;
    end

    for i = 1:numel(plotMethods)
        method = plotMethods(i);

        ax = nexttile(tl, i);
        hold(ax,'on');

        % Simulate the requested method (model vs baseline dispatch)
        sim = local_sim_for_plot(method, P, dt, weights, cfg, theta_star, baselineParams);

        tg = double(sim.t_grid(:));
        yy = double(sim.tau_hist(:));

        % If no measurements exist, derive horizon from simulation duration
        if ~isfinite(xMax)
            xMax = max(tg) + xPad;
        end

        % Clip plotted trajectories to the chosen horizon
        mask = tg <= xMax;
        tg = tg(mask);
        yy = yy(mask);

        hLine = plot(ax, tg, yy, '-', 'LineWidth', 1.8);

        % Door events and measurement overlays are drawn on top of trajectories
        local_plot_door_markers(doorEvents);
        hMeasThis = local_plot_measurements_overlay(ax, t_meas, y_meas, kind_meas, Sth);

        if i == 1
            hSimSrc  = hLine;
            hMeasSrc = hMeasThis(:);
            hMeasSrc = hMeasSrc(isgraphics(hMeasSrc));
        end

        ylim(ax,[0 1]);
        xlim(ax,[0 xMax]);
        grid(ax,'on');
        xlabel(ax,'Time [s]');
        ylabel(ax,'Trust');
        title(ax, sprintf('%s', methodDisplayName(method)), 'Interpreter','none');

        hold(ax,'off');
    end

    % Note panel content: participant ID + wRMSE per plotted method
    lines = strings(0,1);
    lines(end+1) = "Participant: " + string(pid); %#ok<AGROW>
    lines(end+1) = ""; %#ok<AGROW>
    lines(end+1) = "wRMSE"; %#ok<AGROW>

    for i = 1:numel(plotMethods)
        method = plotMethods(i);
        wr = local_lookup_participant_wrmse(wrmseLookup, method, pid);
        lines(end+1) = sprintf("%s : %s", methodDisplayName(method), char(local_fmt_num_or_na(wr))); %#ok<AGROW>
    end

    txtNote = strjoin(cellstr(lines), newline);

    % Build legend proxies inside axLegendHost (legend cannot span tiles reliably)
    proxy = gobjects(0,1);

    % Proxy: simulated trust line (copy style from the first subplot if available)
    if isgraphics(hSimSrc)
        p1 = plot(axLegendHost, NaN, NaN, ...
            'LineStyle', hSimSrc.LineStyle, ...
            'LineWidth', hSimSrc.LineWidth, ...
            'Color',     hSimSrc.Color, ...
            'Marker',    'none', ...
            'DisplayName','Simulated trust');
    else
        p1 = plot(axLegendHost, NaN, NaN, '-', ...
            'LineWidth', 1.8, ...
            'Color', Sth.colors.cyan, ...
            'Marker','none', ...
            'DisplayName','Simulated trust');
    end
    proxy(end+1,1) = p1; %#ok<AGROW>

    % Proxies: measurement markers (one per measurement kind)
    for i = 1:numel(hMeasSrc)
        h = hMeasSrc(i);
        if ~isgraphics(h), continue; end

        pp = plot(axLegendHost, NaN, NaN, ...
            'LineStyle','none', ...
            'Marker', h.Marker, ...
            'MarkerSize', h.MarkerSize, ...
            'LineWidth', h.LineWidth);

        try, pp.MarkerEdgeColor = h.MarkerEdgeColor; end %#ok<TRYNC>
        try, pp.MarkerFaceColor = h.MarkerFaceColor; end %#ok<TRYNC>
        try, pp.DisplayName     = h.DisplayName; end %#ok<TRYNC>

        proxy(end+1,1) = pp; %#ok<AGROW>
    end

    lgd = legend(axLegendHost, proxy, 'Orientation','horizontal', 'Location','north');
    lgd.Box = 'on';
    lgd.FontName = Sth.fontBody;
    lgd.FontSize = Sth.legendFont;
    lgd.Interpreter = Sth.interpMath;

    % Best-effort single-row legend
    try, lgd.NumColumns = numel(proxy); catch, end %#ok<CTCH>

    % Center legend box within its host axes
    lgd.Units = 'normalized';
    lp = lgd.Position;
    lp(1) = 0.5 - lp(3)/2;
    lp(1) = max(0.01, lp(1));
    lgd.Position = lp;

    % Draw note in the sixth tile using legend styling for consistent look
    axNote = nexttile(tl, 6);
    local_add_note_in_empty_panel(axNote, txtNote, Sth, lgd);

    thesisFinalizeFigure(fig, Sth);
    thesisExport(fig, fullfile(outDirPid, sprintf("traj_%s_methods_grid", char(pid))));
end

function local_plot_combo(outDirPid, pid, bestBaseline, constMethods, ...
    P, dt, weights, cfg, theta_star, baselineParams, ...
    t_meas, y_meas, kind_meas, doorEvents, wrmseLookup, Sth)
% local_plot_combo Combo plot: constants (dashed) + model + best baseline + measurements.
%
% Layout:
%   - Left: main trust trajectory plot with overlays
%   - Right-top: legend panel
%   - Right-bottom: participant-specific wRMSE note panel

    fig = figure('Visible','off','Color','w', ...
        'Name',sprintf('Traj %s model_vs_best', pid));
    thesisStyle(fig);

    set(fig,'Units','centimeters');
    set(fig,'Position',[2 2 Sth.figSizeTrajectoryCombo]);

    % Normalized layout for main plot + right-column legend/note
    xL = 0.08;  yB = 0.12;  hAll = 0.80;  gap = 0.02;
    wR = 0.22;  xR = 1 - 0.02 - wR;
    wL = xR - gap - xL;

    yNote = yB;             hNote = 0.40*hAll;
    yLeg  = yB + hNote;     hLeg  = hAll - hNote;

    axPlot = axes(fig,'Units','normalized','Position',[xL yB wL hAll]);
    axLeg  = axes(fig,'Units','normalized','Position',[xR yLeg  wR hLeg ]);
    axNote = axes(fig,'Units','normalized','Position',[xR yNote wR hNote]);

    axis(axLeg,'off');
    axis(axNote,'off');

    % -------------------------
    % Main plot (left)
    % -------------------------
    hold(axPlot,'on');

    % Plot horizon: stop shortly after final measurement marker (if any)
    xPad = 20; % [s]

    tEndMeas = NaN;
    if ~isempty(t_meas)
        tEndMeas = max(double(t_meas(:)));
    end

    xMax = NaN;
    if isfinite(tEndMeas)
        xMax = tEndMeas + xPad;
    end

    constColors = [Sth.colors.cyanSoft1; Sth.colors.blue; Sth.colors.burgundy];
    hConst = gobjects(numel(constMethods),1);

    % Constants (dashed)
    for i = 1:numel(constMethods)
        m = constMethods(i);
        sim = local_sim_for_plot(m, P, dt, weights, cfg, theta_star, baselineParams);

        c = constColors(min(i,size(constColors,1)),:);

        tg = double(sim.t_grid(:));
        yy = double(sim.tau_hist(:));

        if ~isfinite(xMax)
            xMax = max(tg) + xPad;
        end

        mask = tg <= xMax;
        tg = tg(mask);
        yy = yy(mask);

        hConst(i) = plot(axPlot, tg, yy, ...
            '--', 'LineWidth', 1.2, ...
            'Color', c, ...
            'DisplayName', char(methodDisplayName(m)));
    end

    % Model (solid, thicker)
    simM = local_sim_for_plot("simple_selected", P, dt, weights, cfg, theta_star, baselineParams);
    tg = double(simM.t_grid(:));
    yy = double(simM.tau_hist(:));

    if ~isfinite(xMax)
        xMax = max(tg) + xPad;
    end

    mask = tg <= xMax;
    tg = tg(mask);
    yy = yy(mask);

    hModel = plot(axPlot, tg, yy, ...
        '-', 'LineWidth', 2.2, ...
        'Color', Sth.colors.cyan, ...
        'DisplayName', char(methodDisplayName("simple_selected")));

    % Best baseline (solid)
    simB = local_sim_for_plot(bestBaseline, P, dt, weights, cfg, theta_star, baselineParams);
    tg = double(simB.t_grid(:));
    yy = double(simB.tau_hist(:));

    if ~isfinite(xMax)
        xMax = max(tg) + xPad;
    end

    mask = tg <= xMax;
    tg = tg(mask);
    yy = yy(mask);

    hBest = plot(axPlot, tg, yy, ...
        '-', 'LineWidth', 2.0, ...
        'Color', Sth.colors.yellow, ...
        'DisplayName', char(methodDisplayName(bestBaseline)));

    % Door event markers and aligned measurements overlay
    local_plot_door_markers(doorEvents);
    hMeas = local_plot_measurements_overlay(axPlot, t_meas, y_meas, kind_meas, Sth);

    ylim(axPlot,[0 1]);
    xlim(axPlot,[0 xMax]);
    grid(axPlot,'on');
    xlabel(axPlot,'Time [s]');
    ylabel(axPlot,'Trust');
    title(axPlot, sprintf('Participant %s - model + best baseline + constants', pid), 'Interpreter','none');

    hold(axPlot,'off');

    % -------------------------
    % Legend (top-right): desired ordering and spacing
    % -------------------------
    hold(axLeg,'on');

    % Legend ordering:
    %   model, best baseline, constants..., measurement kinds...
    H_lines = [hModel(:); hBest(:); hConst(:)];
    H_lines = H_lines(isgraphics(H_lines));

    H_meas = hMeas(:);
    H_meas = H_meas(isgraphics(H_meas));

    H = [H_lines; H_meas];

    % Build proxies so legend is drawn within axLeg (independent of axPlot)
    proxy = gobjects(numel(H),1);
    for i = 1:numel(H)
        h = H(i);

        ls  = '-';   lw  = 1.0;  c = [0 0 0];
        mk  = 'none'; ms  = 6;
        mec = 'auto'; mfc = 'none';

        try, ls  = h.LineStyle; end %#ok<TRYNC>
        try, lw  = h.LineWidth; end %#ok<TRYNC>
        try, c   = h.Color; end %#ok<TRYNC>
        try, mk  = h.Marker; end %#ok<TRYNC>
        try, ms  = h.MarkerSize; end %#ok<TRYNC>
        try, mec = h.MarkerEdgeColor; end %#ok<TRYNC>
        try, mfc = h.MarkerFaceColor; end %#ok<TRYNC>

        proxy(i) = plot(axLeg, NaN, NaN, ...
            'LineStyle', ls, 'LineWidth', lw, 'Color', c, ...
            'Marker', mk, 'MarkerSize', ms);

        try, proxy(i).MarkerEdgeColor = mec; end %#ok<TRYNC>
        try, proxy(i).MarkerFaceColor = mfc; end %#ok<TRYNC>
        try, proxy(i).DisplayName = h.DisplayName; end %#ok<TRYNC>
    end

    lgd = legend(axLeg, proxy, 'Location','northeast');
    lgd.Box = 'on';
    lgd.FontName = Sth.fontBody;
    lgd.FontSize = Sth.legendFont;

    % Expand legend height for readability
    lgd.Units = 'normalized';
    p = lgd.Position;
    p(4) = min(0.98, p(4)*1.3);
    p(2) = max(0.02, p(2) - (p(4)-lgd.Position(4)));
    lgd.Position = p;

    % Math interpreter for labels containing '$'
    lgd.Interpreter = Sth.interpMath;

    % Best-effort increase in item token height (varies by MATLAB version)
    try
        its = lgd.ItemTokenSize;
        lgd.ItemTokenSize = [its(1), max(its(2), 22)];
    catch
    end

    hold(axLeg,'off');

    % Ensure helper axes do not display ticks or backgrounds
    set([axLeg axNote], ...
        'Visible','off', ...
        'XTick',[], 'YTick',[], ...
        'XColor','none', 'YColor','none', ...
        'Color','none', ...
        'Box','off');

    % -------------------------
    % Note (bottom-right): participant-level wRMSE values
    % -------------------------
    lines = strings(0,1);
    lines(end+1) = "Participant: " + string(pid); %#ok<AGROW>
    lines(end+1) = ""; %#ok<AGROW>
    lines(end+1) = "wRMSE"; %#ok<AGROW>

    wrM = local_lookup_participant_wrmse(wrmseLookup, "simple_selected", pid);
    lines(end+1) = methodDisplayName("simple_selected") + ": " + local_fmt_num_or_na(wrM); %#ok<AGROW>

    wrB = local_lookup_participant_wrmse(wrmseLookup, bestBaseline, pid);
    lines(end+1) = methodDisplayName(bestBaseline) + ": " + local_fmt_num_or_na(wrB); %#ok<AGROW>

    for i = 1:numel(constMethods)
        m = constMethods(i);
        wrC = local_lookup_participant_wrmse(wrmseLookup, m, pid);
        lines(end+1) = methodDisplayName(m) + ": " + local_fmt_num_or_na(wrC); %#ok<AGROW>
    end

    txt = strjoin(cellstr(lines), newline);

    text(axNote, 0.98, 0.02, txt, ...
        'Units','normalized', ...
        'VerticalAlignment','bottom', ...
        'HorizontalAlignment','right', ...
        'FontSize', Sth.fontSizeAx, ...
        'FontName', Sth.fontBody, ...
        'BackgroundColor', lgd.Color, ...
        'EdgeColor', lgd.EdgeColor, ...
        'LineWidth', lgd.LineWidth, ...
        'Margin', 8, ...
        'Interpreter','none');

    thesisFinalizeFigure(fig, Sth);
    thesisExport(fig, fullfile(outDirPid, sprintf("traj_%s_model_vs_bestbaseline", char(pid))));
end

function local_plot_door_markers(doorEvents)
% local_plot_door_markers Plot door-event time markers as vertical dotted lines.
%
% INPUT
%   doorEvents (struct array) - expected field: .t (event time in seconds)

    if isempty(doorEvents), return; end
    if ~isstruct(doorEvents) || ~isfield(doorEvents, "t"), return; end

    dts = [doorEvents.t];
    if isempty(dts), return; end

    yl = [0 1];
    for td = dts(:)'
        hh = plot([td td], yl, ':', 'LineWidth', 0.5);
        set(hh, 'HandleVisibility','off');
    end
end

function sim = local_sim_for_plot(method, P, dt, weights, cfg, theta_star, baselineParams)
% local_sim_for_plot Dispatch simulator for plotting based on method ID.
%
%   - "simple_selected": fitted SIMPLE model (theta_star)
%   - otherwise: baseline simulator with (weights, baselineParams, cfg)
%
% OUTPUT
%   sim (struct) - expected fields: t_grid, tau_hist; may also contain doorEvents

    method = string(method);
    if method == "simple_selected"
        sim = trust_simulate_or_predict_one_participant("simple", theta_star, P, dt);
    else
        sim = trust_simulate_baseline_one_participant(method, P, dt, weights, baselineParams, cfg);
    end

    if ~isfield(sim, "t_grid") || ~isfield(sim, "tau_hist")
        error("[A6] Simulator output missing t_grid/tau_hist for method %s.", method);
    end
end

function [t_meas, y_meas, kind_meas] = local_unpack_measurements(measurements)
% local_unpack_measurements Convert measurement struct array to aligned vectors.
%
% Expected fields per element (if present):
%   measurements(i).t    - scalar time [s]
%   measurements(i).y    - scalar trust measurement (typically in [0,1])
%   measurements(i).kind - label identifying measurement source/type
%
% Non-finite (t,y) entries are removed.

    if isempty(measurements)
        t_meas = zeros(0,1);
        y_meas = zeros(0,1);
        kind_meas = strings(0,1);
        return;
    end

    M = numel(measurements);
    t_meas    = NaN(M,1);
    y_meas    = NaN(M,1);
    kind_meas = strings(M,1);

    for i = 1:M
        if isfield(measurements(i), "t"),    t_meas(i) = double(measurements(i).t); end
        if isfield(measurements(i), "y"),    y_meas(i) = double(measurements(i).y); end
        if isfield(measurements(i), "kind"), kind_meas(i) = string(measurements(i).kind); end
    end

    % Keep only finite (t,y) pairs
    ok = isfinite(t_meas) & isfinite(y_meas);
    t_meas = t_meas(ok);
    y_meas = y_meas(ok);
    kind_meas = kind_meas(ok);
end

function T = local_rename_var(T, oldName, newName)
% local_rename_var Rename a table variable (version-compatible implementation).
%
% INPUTS
%   T (table)
%   oldName (string|char)
%   newName (string|char)
%
% OUTPUT
%   T (table) - updated table

    oldName = char(oldName);
    newName = char(newName);
    vn = T.Properties.VariableNames;
    idx = find(strcmp(vn, oldName), 1);
    if isempty(idx)
        error("local_rename_var: variable '%s' not found.", oldName);
    end
    vn{idx} = newName;
    T.Properties.VariableNames = vn;
end

function hMeas = local_plot_measurements_overlay(ax, t_meas, y_meas, kind_meas, Sth)
% local_plot_measurements_overlay Overlay aligned trust measurements (marker-only).
%
% Measurements are grouped by kind, producing one plotted handle per kind to
% enable one legend entry per measurement source/type.
%
% INPUTS
%   ax        - target axes
%   t_meas    - measurement times [s]
%   y_meas    - measurement values
%   kind_meas - measurement kind labels
%   Sth       - style struct from thesisStyle()
%
% OUTPUT
%   hMeas - graphics handles, one per unique measurement kind (stable order)

    hMeas = gobjects(0,1);

    if isempty(t_meas) || isempty(y_meas) || isempty(ax)
        return;
    end

    t_meas = double(t_meas(:));
    y_meas = double(y_meas(:));
    kind_meas = string(kind_meas(:));

    % Stable order (first appearance) keeps legend ordering consistent
    kinds = unique(kind_meas, 'stable');

    markers = {'o','s','^','d'};

    % Color mapping for measurement kinds (relies on provided style fields)
    measColors = [Sth.colors.green;
                  Sth.colors.purple;
                  Sth.colors.purple;
                  Sth.colors.red];

    hMeas = gobjects(numel(kinds), 1);

    for i = 1:numel(kinds)
        k = kinds(i);
        mask = (kind_meas == k);

        mk = markers{1 + mod(i-1, numel(markers))};
        cc = measColors(1 + mod(i-1, size(measColors,1)), :);

        % One plot call per kind -> one legend entry per kind
        hMeas(i) = plot(ax, t_meas(mask), y_meas(mask), mk, ...
            'LineStyle','none', ...
            'MarkerSize', 6, ...
            'LineWidth', 1.0, ...
            'MarkerEdgeColor', cc, ...
            'MarkerFaceColor', 'none', ...
            'DisplayName', char(measurementDisplayName(k)));
    end
end

function local_add_note_below_legend(fig, lgd, txt, Sth) %#ok<DEFNU>
% local_add_note_below_legend Add an annotation textbox below a legend (best-effort).
%
% This helper is unused in this file but kept for compatibility with the
% existing codebase and for potential reuse.

    try
        lgd.Units = 'normalized';
        p = lgd.Position; % [x y w h] in normalized figure units

        % Place a box below legend, using legend width
        x = p(1) - 0.02;
        w = p(3);
        h = min(0.33, p(2) - 0.02);
        y = max(0.01, p(2) - h - 0.16);

        annotation(fig, 'textbox', [x y w h], ...
            'String', txt, ...
            'Interpreter', 'none', ...
            'FontName', Sth.fontBody, ...
            'FontSize', Sth.fontSizeAx, ...
            'EdgeColor', [0.3 0.3 0.3], ...
            'BackgroundColor', 'w', ...
            'VerticalAlignment', 'top', ...
            'HorizontalAlignment', 'left', ...
            'FitBoxToText', 'off');
    catch
    end
end

function local_add_note_in_empty_panel(ax, txt, Sth, lgd)
% local_add_note_in_empty_panel Add a note box inside an "empty" tile (axes off).
%
% The note is placed using normalized coordinates so its placement does not
% depend on axis limits. The legend's styling is copied where available to
% maintain consistent visual formatting.

    try
        axis(ax,'off');

        % Position within tile (normalized axes coordinates)
        x = 0.87;
        y = -0.1;
        ha = 'right';

        % Copy legend styling for consistent visual language across figures
        edgeCol = [0.3 0.3 0.3];
        bgCol   = 'w';
        lw      = 0.5;
        try, edgeCol = lgd.EdgeColor; end %#ok<TRYNC>
        try, bgCol   = lgd.Color;     end %#ok<TRYNC>
        try, lw      = lgd.LineWidth; end %#ok<TRYNC>

        text(ax, x, y, txt, ...
            'Units','normalized', ...
            'VerticalAlignment','bottom', ...
            'HorizontalAlignment', ha, ...
            'FontSize', Sth.fontSizeAx, ...
            'FontName', Sth.fontBody, ...
            'BackgroundColor', bgCol, ...
            'EdgeColor', edgeCol, ...
            'LineWidth', lw, ...
            'Margin', 8, ...
            'Interpreter','none');
    catch
        % Best-effort helper; plotting should continue even if note fails.
    end
end

function s = local_fmt_num_or_na(x)
% local_fmt_num_or_na Format numeric value as fixed precision; return "NA" if non-finite.

    if ~isfinite(x)
        s = "NA";
    else
        s = sprintf("%.4f", x);
        s = string(s);
    end
end
