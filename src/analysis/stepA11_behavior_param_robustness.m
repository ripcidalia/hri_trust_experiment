function stepA11_behavior_param_robustness(run_id, varargin)
% stepA11_behavior_param_robustness Robustness analysis for per-participant behavioral parameters.
%
% This step evaluates the robustness/identifiability of per-participant
% behavioral model parameters estimated on VALID data (mechanistic analysis).
% It performs two complementary analyses:
%
%   A11.1 Random split stability (primary)
%     For each VALID participant and for each selected model (0/1/2), repeat S times:
%       - Randomly split doors into a fit subset (SplitN) and a held-out subset
%       - Fit parameters on the fit subset (Model 1/2) and predict on held-out
%       - Record parameter samples (when applicable)
%       - Record held-out NLL and held-out Brier score
%       - Record Pearson correlation corr(p_holdout, y_holdout) on held-out only
%
%   A11.2 Blockwise stability (qualitative add-on)
%     For each VALID participant:
%       - Use the A10-selected best model
%       - Fit the best model separately on blocks 1/2/3 (doors 1–20, 21–40, 41–60)
%       - Assign a simple drift label (stable/drifting/under_identified)
%       - Save a per-participant plot of blockwise best-model parameters
%
% INPUTS (required)
%   run_id (string|char)
%       Analysis run identifier. Used to locate:
%         A7 VALID dataset:
%           derived/analysis_runs/<run_id>/stepA7_behavior_dataset/behavior_dataset_valid.mat
%         A10 results:
%           derived/analysis_runs/<run_id>/stepA10_behavior_fit_by_participant/A10_params_by_participant.mat
%
% NAME-VALUE ARGUMENTS (optional)
%   "OutDir" (string|char)
%       Output directory. Default:
%         derived/analysis_runs/<run_id>/stepA11_behavior_param_robustness
%
%   "Overwrite" (logical scalar)
%       If false (default), error if outputs already exist.
%
%   "S" (numeric scalar)
%       Number of random splits for A11.1 (default: 1000).
%
%   "SplitN" (numeric scalar)
%       Size of fit subset per split (default: 30). Held-out size is N - SplitN.
%
%   "RandomSeed" (numeric scalar)
%       RNG seed used for random splits (default: 1).
%
%   "Models" (numeric vector)
%       Which models to evaluate in A11.1 using indices [0 1 2] (default: [0 1 2]).
%
%   "EpsMax" (numeric scalar)
%       Maximum lapse probability for Model 2 (default: 0.5).
%
%   "KHuge" (numeric scalar)
%       Diagnostic threshold for "huge" k estimates (default: 100).
%
%   "MinHeldoutVarP" (numeric scalar)
%       If var(p_holdout) < threshold, correlation is treated as undefined (default: 1e-6).
%
%   "MinHeldoutVarY" (numeric scalar)
%       If var(y_holdout) < threshold, correlation is treated as undefined (default: 1e-6).
%
% OUTPUTS
%   (none)
%       Writes artifacts to:
%         derived/analysis_runs/<run_id>/stepA11_behavior_param_robustness/
%           - A11_split_stability.csv
%           - A11_split_stability.mat          (Tsplits + splitDetails + meta)
%           - A11_blockwise_params.csv
%           - A11_blockwise_params.mat         (Tblocks + blockDetails)
%           - meta.mat / meta.json
%           - figures/*.png
%           - figures/blockwise_by_participant/pid_<ID>_blockwise_bestmodel.png
%
% REQUIRED TABLE COLUMNS (from A7 behavior dataset)
%   participant_id, door_index, block_index,
%   tau_decision, self_confidence, sc_centered, margin_treshold,
%   followed, is_valid_label
%
% DEPENDENCIES (assumed on MATLAB path)
%   must_exist_file, ensure_dir, save_json, thesisStyle, thesisFinalizeFigure, thesisExport

    % ------------------------------------------------------------------
    % Parse required input and name-value arguments
    % ------------------------------------------------------------------
    if nargin < 1 || isempty(run_id)
        error("stepA11_behavior_param_robustness: run_id is required.");
    end
    run_id = string(run_id);

    p = inputParser;
    p.addParameter("OutDir", "", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("S", 1000, @(x) isnumeric(x) && isscalar(x) && x>=1);
    p.addParameter("SplitN", 30, @(x) isnumeric(x) && isscalar(x) && x>=1);
    p.addParameter("RandomSeed", 1, @(x) isnumeric(x) && isscalar(x));
    p.addParameter("Models", [0 1 2], @(x) isnumeric(x) && isvector(x));
    p.addParameter("EpsMax", 0.5, @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
    p.addParameter("KHuge", 100, @(x) isnumeric(x) && isscalar(x) && x>0);
    p.addParameter("MinHeldoutVarP", 1e-6, @(x) isnumeric(x) && isscalar(x) && x>=0);
    p.addParameter("MinHeldoutVarY", 1e-6, @(x) isnumeric(x) && isscalar(x) && x>=0);
    p.parse(varargin{:});
    args = p.Results;

    rng(args.RandomSeed);

    % Thesis plotting defaults + style struct
    S = thesisStyle();

    % ------------------------------------------------------------------
    % Load A7 VALID dataset (behavior table)
    % ------------------------------------------------------------------
    a7Dir = fullfile("derived","analysis_runs",run_id,"stepA7_behavior_dataset");
    validMat = fullfile(a7Dir, "behavior_dataset_valid.mat");
    must_exist_file(validMat, "A7 VALID dataset");

    Sv = load(validMat, "T");
    if ~isfield(Sv,"T") || ~istable(Sv.T)
        error("[A11] VALID mat missing table T.");
    end
    T = Sv.T;

    % Validate data contract with Step A7
    reqCols = ["participant_id","door_index","block_index", ...
               "tau_decision","self_confidence","sc_centered","margin_treshold", ...
               "followed","is_valid_label"];
    assert(all(ismember(reqCols, string(T.Properties.VariableNames))), ...
        "[A11] A7 VALID missing required columns.");

    % Keep analysis subset and remove NaNs in predictors used by models
    T = T(T.is_valid_label==1, :);
    okPred = isfinite(T.tau_decision) & isfinite(T.self_confidence) & ...
             isfinite(T.sc_centered) & isfinite(T.margin_treshold);
    T = T(okPred, :);

    % ------------------------------------------------------------------
    % Load A10 results (best model per participant)
    % ------------------------------------------------------------------
    a10Dir = fullfile("derived","analysis_runs",run_id,"stepA10_behavior_fit_by_participant");
    a10Mat = fullfile(a10Dir, "A10_params_by_participant.mat");
    must_exist_file(a10Mat, "A10_params_by_participant.mat");

    Sa10 = load(a10Mat, "Tsum");
    if ~isfield(Sa10,"Tsum") || ~istable(Sa10.Tsum)
        error("[A11] A10 mat missing table Tsum.");
    end
    Ta10 = Sa10.Tsum;

    if ~ismember("participant_id", string(Ta10.Properties.VariableNames)) || ...
       ~ismember("best_model_idx", string(Ta10.Properties.VariableNames))
        error("[A11] A10 Tsum must contain participant_id and best_model_idx.");
    end

    % ------------------------------------------------------------------
    % Output directories and overwrite guard
    % ------------------------------------------------------------------
    outDir = string(args.OutDir);
    if strlength(outDir)==0
        outDir = fullfile("derived","analysis_runs",run_id,"stepA11_behavior_param_robustness");
    end
    ensure_dir(outDir);

    figDir = fullfile(outDir, "figures");
    ensure_dir(figDir);

    % Per-participant blockwise plots directory
    figByPidDir = fullfile(figDir, "blockwise_by_participant");
    ensure_dir(figByPidDir);

    splitCsv = fullfile(outDir, "A11_split_stability.csv");
    splitMat = fullfile(outDir, "A11_split_stability.mat");
    blockCsv = fullfile(outDir, "A11_blockwise_params.csv");
    blockMat = fullfile(outDir, "A11_blockwise_params.mat");
    metaMat  = fullfile(outDir, "meta.mat");
    metaJson = fullfile(outDir, "meta.json");

    if ~args.Overwrite
        if isfile(splitCsv) || isfile(splitMat) || isfile(blockCsv) || isfile(blockMat)
            error("[A11] Outputs exist. Set Overwrite=true to replace. (%s)", outDir);
        end
    end

    % ------------------------------------------------------------------
    % Participants present in A7 VALID
    % ------------------------------------------------------------------
    pid_all = string(T.participant_id);
    uniqP = unique(pid_all);
    nP = numel(uniqP);

    models = unique(round(args.Models(:)'));
    Ssplits = double(args.S);
    splitN = double(args.SplitN);
    epsMax = double(args.EpsMax);

    % ------------------------------------------------------------------
    % A11.1 Random split stability
    % ------------------------------------------------------------------
    Tsplits = table();
    Tsplits.participant_id = strings(0,1);
    Tsplits.model = strings(0,1);
    Tsplits.N = zeros(0,1);
    Tsplits.S = zeros(0,1);
    Tsplits.splitN = zeros(0,1);

    % Parameter stability (across random splits)
    Tsplits.param_sd_k = NaN(0,1);
    Tsplits.param_mad_k = NaN(0,1);
    Tsplits.param_sd_beta = NaN(0,1);
    Tsplits.param_mad_beta = NaN(0,1);
    Tsplits.param_sd_eps = NaN(0,1);
    Tsplits.param_mad_eps = NaN(0,1);
    Tsplits.fit_fail_rate = NaN(0,1);

    % Held-out robustness (prediction on held-out split)
    Tsplits.nll_holdout_mean = NaN(0,1);
    Tsplits.nll_holdout_sd   = NaN(0,1);
    Tsplits.brier_holdout_mean = NaN(0,1);
    Tsplits.brier_holdout_sd   = NaN(0,1);
    Tsplits.py_corr_holdout_mean = NaN(0,1);
    Tsplits.py_corr_holdout_sd   = NaN(0,1);

    % Flags / diagnostics
    Tsplits.flag_low_N = false(0,1);
    Tsplits.flag_k_huge = false(0,1);
    Tsplits.flag_corr_undefined_rate = NaN(0,1);

    % Optional detailed storage (kept small): per participant+model arrays
    splitDetails = struct();
    splitDetails.run_id = char(run_id);
    splitDetails.entries = {};

    fprintf("[A11.1] Random split stability: S=%d, split=%d/%d\n", Ssplits, splitN, splitN);

    for pi = 1:nP
        pid = uniqP(pi);
        Tp = T(pid_all == pid, :);

        % Preserve door order (not required for random splits, but improves traceability)
        [~,ord] = sort(double(Tp.door_index(:)));
        Tp = Tp(ord,:);

        % Observations and predictors
        y   = double(Tp.followed(:));
        tau = double(Tp.tau_decision(:));
        m   = double(Tp.margin_treshold(:));
        scC = double(Tp.sc_centered(:));
        N   = numel(y);

        % Guard: need at least SplitN for fit and SplitN for held-out in every split
        if N < 2*splitN
            warning("[A11.1] pid=%s has N=%d < %d required for %d/%d. Skipping.", ...
                pid, N, 2*splitN, splitN, splitN);
            continue;
        end

        for mi = 1:numel(models)
            midx = models(mi);
            mname = model_name_from_idx(midx);

            % Per-split parameter samples (NaN when not applicable or failed)
            k_s    = NaN(Ssplits,1);
            beta_s = NaN(Ssplits,1);
            eps_s  = NaN(Ssplits,1);

            % Per-split held-out metrics (NaN when fit failed)
            nll_ho  = NaN(Ssplits,1);
            br_ho   = NaN(Ssplits,1);
            corr_ho = NaN(Ssplits,1);

            fitFail = 0;
            corrUndef = 0;

            for s = 1:Ssplits
                % Random fit subset A of size splitN; held-out is B = complement
                idxA = randperm(N, splitN);
                maskA = false(N,1); maskA(idxA) = true;
                idxB = find(~maskA);

                % ----------------------------
                % Fit on A (Model 0: no parameters)
                % ----------------------------
                okFit = true;
                switch midx
                    case 0
                        % No fit: Model 0 uses tau_decision directly.

                    case 1
                        % k >= 0 via k = exp(z)
                        z0_1 = log(10);
                        f1 = @(z) nll_model1(exp(z), m(maskA), y(maskA));
                        [khat, ~, ok] = safe_fit_1d_exp(f1, z0_1);
                        okFit = ok;
                        if okFit
                            k_s(s) = khat;
                        end

                    case 2
                        % k >= 0 via exp(z1); eps in [0, epsMax] via epsMax*sigmoid(z3)
                        z0_2 = [log(10); 0; logit(0.05/epsMax)];
                        f2 = @(z) nll_model2(exp(z(1)), z(2), epsMax*sigmoid(z(3)), ...
                                             tau(maskA), scC(maskA), y(maskA));
                        [khat, betahat, epshat, ~, ok] = safe_fit_model2(f2, z0_2, epsMax);
                        okFit = ok;
                        if okFit
                            k_s(s) = khat;
                            beta_s(s) = betahat;
                            eps_s(s) = epshat;
                        end

                    otherwise
                        error("[A11.1] Unknown model idx: %d", midx);
                end

                if ~okFit
                    fitFail = fitFail + 1;
                    continue; % leave held-out metrics NaN for this split
                end

                % ----------------------------
                % Predict on held-out B only
                % ----------------------------
                yB = y(idxB);

                switch midx
                    case 0
                        pB = clamp01(tau(idxB));

                    case 1
                        pB = sigmoid(k_s(s) .* m(idxB));

                    case 2
                        zB = k_s(s).*tau(idxB) + beta_s(s).*scC(idxB);
                        pB = (1-eps_s(s)).*sigmoid(zB) + eps_s(s).*0.5;
                        pB = clamp01(pB);
                end

                % Held-out scores
                nll_ho(s) = bernoulli_nll(yB, pB);
                br_ho(s)  = mean((pB - yB).^2);

                % Held-out Pearson correlation corr(p_holdout, y_holdout)
                if var(pB) < args.MinHeldoutVarP || var(yB) < args.MinHeldoutVarY
                    corrUndef = corrUndef + 1;
                    corr_ho(s) = NaN;
                else
                    corr_ho(s) = corr(pB(:), yB(:), 'Type','Pearson');
                end
            end

            % Summaries over splits
            failRate = fitFail / max(1,Ssplits);
            corrUndefRate = corrUndef / max(1,Ssplits);

            row = table();
            row.participant_id = pid;
            row.model = mname;
            row.N = N;
            row.S = Ssplits;
            row.splitN = splitN;

            % Parameter stability summaries (only where applicable)
            if midx == 1 || midx == 2
                row.param_sd_k = std(k_s, 'omitnan');
                row.param_mad_k = mad(k_s, 1); % median absolute deviation
                row.flag_k_huge = any(k_s(isfinite(k_s)) > args.KHuge);
            else
                row.param_sd_k = NaN;
                row.param_mad_k = NaN;
                row.flag_k_huge = false;
            end

            if midx == 2
                row.param_sd_beta = std(beta_s, 'omitnan');
                row.param_mad_beta = mad(beta_s, 1);
                row.param_sd_eps = std(eps_s, 'omitnan');
                row.param_mad_eps = mad(eps_s, 1);
            else
                row.param_sd_beta = NaN; row.param_mad_beta = NaN;
                row.param_sd_eps  = NaN; row.param_mad_eps  = NaN;
            end

            row.fit_fail_rate = failRate;

            row.nll_holdout_mean = mean(nll_ho, 'omitnan');
            row.nll_holdout_sd   = std(nll_ho, 'omitnan');
            row.brier_holdout_mean = mean(br_ho, 'omitnan');
            row.brier_holdout_sd   = std(br_ho, 'omitnan');
            row.py_corr_holdout_mean = mean(corr_ho, 'omitnan');
            row.py_corr_holdout_sd   = std(corr_ho, 'omitnan');
            row.flag_corr_undefined_rate = corrUndefRate;

            % Informational flag (pipeline typically yields N~60)
            row.flag_low_N = (N < 60);

            Tsplits = [Tsplits; row]; %#ok<AGROW>

            % Store split-level arrays (run-local diagnostics; can be large for S=1000)
            E = struct();
            E.participant_id = char(pid);
            E.model_idx = midx;
            E.model = char(mname);
            E.N = N; E.S = Ssplits; E.splitN = splitN;
            E.k = k_s;
            E.beta = beta_s;
            E.eps = eps_s;
            E.nll_holdout = nll_ho;
            E.brier_holdout = br_ho;
            E.py_corr_holdout = corr_ho;
            E.fit_fail_rate = failRate;
            E.corr_undefined_rate = corrUndefRate;
            splitDetails.entries{end+1,1} = E; %#ok<AGROW>
        end
    end

    writetable(Tsplits, splitCsv);

    % ------------------------------------------------------------------
    % A11.2 Blockwise stability using A10-selected best model
    % ------------------------------------------------------------------
    Tblocks = table();
    Tblocks.participant_id = strings(0,1);
    Tblocks.best_model_idx = NaN(0,1);
    Tblocks.best_model = strings(0,1);

    % Parameters by block (NaN when not applicable to the best model)
    Tblocks.k_b1 = NaN(0,1);    Tblocks.k_b2 = NaN(0,1);    Tblocks.k_b3 = NaN(0,1);
    Tblocks.beta_b1 = NaN(0,1); Tblocks.beta_b2 = NaN(0,1); Tblocks.beta_b3 = NaN(0,1);
    Tblocks.eps_b1 = NaN(0,1);  Tblocks.eps_b2 = NaN(0,1);  Tblocks.eps_b3 = NaN(0,1);

    Tblocks.nll_b1 = NaN(0,1);  Tblocks.nll_b2 = NaN(0,1);  Tblocks.nll_b3 = NaN(0,1);
    Tblocks.N_b1 = NaN(0,1);    Tblocks.N_b2 = NaN(0,1);    Tblocks.N_b3 = NaN(0,1);

    Tblocks.drift_label = strings(0,1);          % stable/drifting/under_identified
    Tblocks.drift_score = NaN(0,1);              % magnitude score (heuristic)
    Tblocks.flag_under_identified = false(0,1);

    blockDetails = struct();
    blockDetails.run_id = char(run_id);
    blockDetails.entries = {};

    fprintf("[A11.2] Blockwise stability (best model from A10)\n");

    for pi = 1:nP
        pid = uniqP(pi);

        % Best model index from A10; A10 uses 1..3 corresponding to Models 0..2
        idxA10 = find(string(Ta10.participant_id)==pid, 1);
        if isempty(idxA10)
            warning("[A11.2] pid=%s not found in A10 table. Skipping.", pid);
            continue;
        end
        bestIdx_a10 = double(Ta10.best_model_idx(idxA10));  % 1..3
        bestIdx = bestIdx_a10 - 1;                          % convert to 0..2

        Tp = T(pid_all == pid, :);
        [~,ord] = sort(double(Tp.door_index(:)));
        Tp = Tp(ord,:);

        y   = double(Tp.followed(:));
        tau = double(Tp.tau_decision(:));
        m   = double(Tp.margin_treshold(:));
        scC = double(Tp.sc_centered(:));
        blk = double(Tp.block_index(:));

        row = table();
        row.participant_id = pid;
        row.best_model_idx = bestIdx;
        row.best_model = model_name_from_idx(bestIdx);

        % Fit best model separately on each block
        [p1, n1, ok1] = fit_model_on_block(bestIdx, blk, 1, y, tau, m, scC, epsMax);
        [p2, n2, ok2] = fit_model_on_block(bestIdx, blk, 2, y, tau, m, scC, epsMax);
        [p3, n3, ok3] = fit_model_on_block(bestIdx, blk, 3, y, tau, m, scC, epsMax);

        row.N_b1 = n1; row.N_b2 = n2; row.N_b3 = n3;

        % Populate per-block parameter values and NLL (NaN when not applicable)
        row.k_b1 = p1.k; row.k_b2 = p2.k; row.k_b3 = p3.k;
        row.beta_b1 = p1.beta; row.beta_b2 = p2.beta; row.beta_b3 = p3.beta;
        row.eps_b1  = p1.eps;  row.eps_b2  = p2.eps;  row.eps_b3  = p3.eps;

        row.nll_b1 = p1.nll; row.nll_b2 = p2.nll; row.nll_b3 = p3.nll;

        % Basic identifiability rule: all block fits succeed and have enough samples
        okAll = ok1 && ok2 && ok3 && all([n1 n2 n3] >= 10);
        row.flag_under_identified = ~okAll;

        % Drift label/score heuristic for interpretability
        [label, score] = drift_label_from_blocks(bestIdx, [p1 p2 p3], okAll);
        row.drift_label = label;
        row.drift_score = score;

        Tblocks = [Tblocks; row]; %#ok<AGROW>

        % Store per-participant blockwise fit details
        E = struct();
        E.participant_id = char(pid);
        E.best_model_idx = bestIdx;
        E.best_model = char(row.best_model);
        E.blocks = {p1, p2, p3};
        E.N_blocks = [n1 n2 n3];
        E.ok_blocks = [ok1 ok2 ok3];
        E.drift_label = char(label);
        E.drift_score = score;
        blockDetails.entries{end+1,1} = E; %#ok<AGROW>
    end

    writetable(Tblocks, blockCsv);

    % ------------------------------------------------------------------
    % Per-participant blockwise plots (best-model parameters only)
    % ------------------------------------------------------------------
    for i = 1:height(Tblocks)
        pid = string(Tblocks.participant_id(i));
        pdfPath = fullfile(figByPidDir, sprintf("pid_%s_blockwise_bestmodel", char(pid)));
        make_fig_blockwise_one_participant(pdfPath, Tblocks(i,:), S);
    end

    % ------------------------------------------------------------------
    % Save MAT bundles + meta
    % ------------------------------------------------------------------
    meta = struct();
    meta.run_id = char(run_id);
    meta.created = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    meta.valid_participants = nP;
    meta.S = Ssplits;
    meta.splitN = splitN;
    meta.models = models;
    meta.random_seed = args.RandomSeed;
    meta.epsMax = epsMax;
    meta.heldout_corr_definition = "Pearson corr(p_follow_holdout, y_follow_holdout) on held-out doors only";
    meta.k_huge_threshold = args.KHuge;

    save(splitMat, "Tsplits", "splitDetails", "meta", "-v7.3");
    save(blockMat, "Tblocks", "blockDetails", "-v7.3");
    save(metaMat, "meta");
    save_json(metaJson, meta);

    % ------------------------------------------------------------------
    % Figures (aggregate summaries)
    % ------------------------------------------------------------------
    make_fig_split_param_sd(fullfile(figDir, "split_param_sd_model1_k"), Tsplits, ...
        "model1_threshold", "param_sd_k", ...
        "Split stability: $\sigma(k)$ (thresholded reliance)", S);

    make_fig_split_param_sd(fullfile(figDir, "split_param_sd_model2_k"), Tsplits, ...
        "model2_offset_lapse", "param_sd_k", ...
        "Split stability: $\sigma(k)$ (threshold + modulation + lapse)", S);

    make_fig_split_param_sd(fullfile(figDir, "split_param_sd_model2_beta"), Tsplits, ...
        "model2_offset_lapse", "param_sd_beta", ...
        "Split stability: $\sigma(\alpha)$ (threshold + modulation + lapse)", S);

    make_fig_split_param_sd(fullfile(figDir, "split_param_sd_model2_eps"), Tsplits, ...
        "model2_offset_lapse", "param_sd_eps", ...
        "Split stability: $\sigma(\varepsilon)$ (threshold + modulation + lapse)", S);

    make_fig_blockwise_params(fullfile(figDir, "blockwise_params_bestmodel.png"), Tblocks, S);

    fprintf("[Step A11] Done.\n");
    fprintf("  Output dir: %s\n", outDir);
    fprintf("  Wrote: %s\n", splitCsv);
    fprintf("  Wrote: %s\n", blockCsv);
    fprintf("  Wrote per-participant plots to: %s\n", figByPidDir);
end

% ======================================================================
% Blockwise fit helper
% ======================================================================
function [par, Nb, ok] = fit_model_on_block(modelIdx, blk, b, y, tau, m, scC, epsMax)
% fit_model_on_block Fit a specified model to a single participant block.
%
% INPUTS
%   modelIdx - model index in {0,1,2}
%   blk      - vector of block indices (1..3) for each door
%   b        - target block index to fit (1/2/3)
%   y        - binary follow decisions (1=follow, 0=override)
%   tau      - trust predictor (tau_decision)
%   m        - margin predictor (margin_treshold)
%   scC      - centered self-confidence predictor (sc_centered)
%   epsMax   - maximum lapse for Model 2
%
% OUTPUTS
%   par - struct with fields k, beta, eps, nll (NaN where not applicable)
%   Nb  - number of doors in the block
%   ok  - true if fit produced finite values (model0 always ok if Nb>=2)

    mask = (blk==b);
    Nb = sum(mask);

    par = struct("k",NaN,"beta",NaN,"eps",NaN,"nll",NaN);

    if Nb < 2
        ok = false;
        return;
    end

    yb = y(mask);
    taub = tau(mask);
    mb = m(mask);
    scCb = scC(mask);

    ok = true;

    switch modelIdx
        case 0
            % Model 0: no fitted parameters; evaluate NLL using tau as probability
            pb = clamp01(taub);
            par.nll = bernoulli_nll(yb, pb);

        case 1
            % Model 1: fit k >= 0 (k = exp(z))
            z0_1 = log(10);
            f1 = @(z) nll_model1(exp(z), mb, yb);
            [khat, nllhat, ok1] = safe_fit_1d_exp(f1, z0_1);
            ok = ok1;
            par.k = khat;
            par.nll = nllhat;

        case 2
            % Model 2: fit k >= 0 and eps in [0, epsMax] via transforms
            z0_2 = [log(10); 0; logit(0.05/epsMax)];
            f2 = @(z) nll_model2(exp(z(1)), z(2), epsMax*sigmoid(z(3)), taub, scCb, yb);
            [khat, betahat, epshat, nllhat, ok2] = safe_fit_model2(f2, z0_2, epsMax);
            ok = ok2;
            par.k = khat;
            par.beta = betahat;
            par.eps = epshat;
            par.nll = nllhat;

        otherwise
            error("[A11] Unknown modelIdx in blockwise fit: %d", modelIdx);
    end
end

% ======================================================================
% Drift labeling (exploratory heuristic)
% ======================================================================
function [label, score] = drift_label_from_blocks(modelIdx, pars3, okAll)
% drift_label_from_blocks Assign a qualitative drift label from 3 blockwise fits.
%
% INPUTS
%   modelIdx - model index in {0,1,2}
%   pars3    - 1x3 struct array with fields k, beta, eps, nll
%   okAll    - true if all block fits are considered identified/successful
%
% OUTPUTS
%   label - "stable" | "drifting" | "under_identified"
%   score - heuristic magnitude score (relative/normalized), NaN if under-identified

    % pars3: 1x3 struct array (k,beta,eps,nll)
    if ~okAll
        label = "under_identified";
        score = NaN;
        return;
    end

    switch modelIdx
        case 0
            % Model 0 has no fitted parameters; treat as stable by definition
            label = "stable";
            score = 0;

        case 1
            % Drift heuristic based on relative range of k across blocks
            ks = [pars3(1).k pars3(2).k pars3(3).k];
            km = median(ks, 'omitnan');
            score = (max(ks)-min(ks)) / max(1e-12, km); % relative range
            if score <= 0.3
                label = "stable";
            else
                label = "drifting";
            end

        case 2
            % Drift heuristic combining k (relative), beta (absolute), eps (absolute)
            ks = [pars3(1).k pars3(2).k pars3(3).k];
            bs = [pars3(1).beta pars3(2).beta pars3(3).beta];
            es = [pars3(1).eps pars3(2).eps pars3(3).eps];

            kscore = (max(ks)-min(ks)) / max(1e-12, median(ks,'omitnan'));
            bscore = (max(bs)-min(bs)); % absolute range
            escore = (max(es)-min(es)); % absolute range
            score = max([kscore, bscore/5, escore/0.2]); % normalized combination

            if kscore <= 0.3 && bscore <= 2.0 && escore <= 0.1
                label = "stable";
            else
                label = "drifting";
            end

        otherwise
            label = "under_identified";
            score = NaN;
    end
end

% ======================================================================
% Plot helpers
% ======================================================================
function make_fig_split_param_sd(pathPdf, Tsplits, modelName, fieldName, titleStr, S)
% make_fig_split_param_sd Histogram of a split-stability summary statistic.
%
% INPUTS
%   pathPdf   - output base path for thesisExport (extension handled there)
%   Tsplits   - table from A11.1 (one row per participant+model)
%   modelName - model identifier string (matches Tsplits.model)
%   fieldName - variable name in Tsplits to plot (e.g., "param_sd_k")
%   titleStr  - figure title string
%   S         - thesis style struct

    f = figure('Visible','off');
    thesisStyle(f);
    grid on; hold on;

    M = Tsplits(string(Tsplits.model)==string(modelName), :);
    if isempty(M)
        title([titleStr " (no rows)"], 'Interpreter','none');
        thesisFinalizeFigure(f, S);
        thesisExport(f, string(pathPdf));
        return;
    end

    x = M.(fieldName);
    x = x(isfinite(x));
    if isempty(x)
        title([titleStr " (no finite values)"], 'Interpreter','none');
        saveas(f, pathPdf);
        close(f);
        return;
    end

    histogram(x, 20);
    xlabel(paramDisplayName(fieldName), 'Interpreter','none');
    ylabel('Count');
    title(titleStr, 'Interpreter','none');
    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function make_fig_blockwise_params(pathPdf, Tblocks, S)
% make_fig_blockwise_params Plot blockwise k trajectories for all participants.
%
% This plot overlays k trajectories across blocks (B1/B2/B3) for each participant
% using the A10-selected best model fits stored in Tblocks. For participants whose
% best model has no k (not expected here) or yields NaNs, the trajectory is skipped.

    f = figure('Visible','off');
    thesisStyle(f);
    grid on; hold on;

    if isempty(Tblocks)
        title("A11.2 Blockwise params (no rows)", 'Interpreter','none');
        thesisFinalizeFigure(f, S);
        thesisExport(f, string(pathPdf));
        return;
    end

    x = [1 2 3];
    for i = 1:height(Tblocks)
        k = [Tblocks.k_b1(i) Tblocks.k_b2(i) Tblocks.k_b3(i)];
        if all(~isfinite(k)), continue; end
        plot(x, k, '-o', 'LineWidth', 1.0);
    end

    set(gca,'XTick',x,'XTickLabel',{'B1','B2','B3'});
    xlabel('Block');
    ylabel('k (if applicable)');
    title("Blockwise k trajectories (best model per participant)", 'Interpreter','none');

    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

function make_fig_blockwise_one_participant(pathPdf, row, S)
% make_fig_blockwise_one_participant Plot blockwise best-model parameters for one participant.
%
% Option A: parameter traces only (k/beta/eps where applicable). If the best model
% has no parameters (Model 0), a placeholder plot is still exported.

    f = figure('Visible','off');
    thesisStyle(f);
    grid on; hold on;

    x = [1 2 3];
    bestModel = string(row.best_model);
    pid = string(row.participant_id);
    driftLabel = string(row.drift_label);

    % Extract blockwise parameters (may be NaN depending on best model)
    k = [row.k_b1 row.k_b2 row.k_b3];
    b = [row.beta_b1 row.beta_b2 row.beta_b3];
    e = [row.eps_b1 row.eps_b2 row.eps_b3];

    anyPlotted = false;

    if any(isfinite(k))
        plot(x, k, '-o', 'LineWidth', 1.5, 'DisplayName', 'k');
        anyPlotted = true;
    end
    if any(isfinite(b))
        plot(x, b, '-s', 'LineWidth', 1.5, 'DisplayName', 'beta');
        anyPlotted = true;
    end
    if any(isfinite(e))
        plot(x, e, '-d', 'LineWidth', 1.5, 'DisplayName', 'eps');
        anyPlotted = true;
    end

    set(gca,'XTick',x,'XTickLabel',{'B1','B2','B3'});
    xlabel('Block');

    if ~anyPlotted
        % Model 0 has no parameters; export a minimal placeholder figure
        plot(x, [0 0 0], 'LineWidth', 0.5, 'HandleVisibility','off');
        ylabel('(no parameters for Model 0)');
    else
        ylabel('Parameter value');
        legend('Location','best');
    end

    ttl = sprintf("Blockwise params | pid=%s | %s | %s", char(pid), char(bestModel), char(driftLabel));
    title(ttl, 'Interpreter','none');

    thesisFinalizeFigure(f, S);
    thesisExport(f, string(pathPdf));
end

% ======================================================================
% Model name mapping
% ======================================================================
function name = model_name_from_idx(modelIdx)
% model_name_from_idx Map model index in {0,1,2} to stable identifier string.
    switch modelIdx
        case 0, name = "model0_trust_as_probability";
        case 1, name = "model1_threshold";
        case 2, name = "model2_offset_lapse";
        otherwise, name = "unknown";
    end
end

% ======================================================================
% Likelihoods and math helpers (consistent with A10/A8)
% ======================================================================
function nll = nll_model1(k, margin, y_follow)
% nll_model1 Negative log-likelihood for Model 1: p = sigmoid(k * margin), k >= 0.
    if ~isfinite(k) || k < 0, nll = Inf; return; end
    p = sigmoid(k .* margin);
    nll = bernoulli_nll(y_follow, p);
end

function nll = nll_model2(k, beta, eps, tau, scC, y_follow)
% nll_model2 Negative log-likelihood for Model 2 with lapse:
%   z = k*tau + beta*sc_centered
%   p = (1-eps)*sigmoid(z) + eps*0.5
    if ~isfinite(k) || k < 0 || ~isfinite(beta) || ~isfinite(eps) || eps < 0 || eps > 1
        nll = Inf; return;
    end
    z = k .* tau + beta .* scC;
    pstar = sigmoid(z);
    p = (1-eps).*pstar + eps.*0.5;
    nll = bernoulli_nll(y_follow, p);
end

function nll = bernoulli_nll(y, p)
% bernoulli_nll Bernoulli negative log-likelihood with probability clipping.
    p = min(max(p, 1e-12), 1-1e-12);
    nll = -sum(y .* log(p) + (1-y).*log(1-p));
end

function [k_hat, nll_hat, ok] = safe_fit_1d_exp(f, z0)
% safe_fit_1d_exp Fit a 1D objective with positivity enforced via k = exp(z).
    ok = true;
    try
        zhat = fminsearch(f, z0, optimset('Display','off'));
        k_hat = exp(zhat);
        nll_hat = f(zhat);
        if ~isfinite(k_hat) || ~isfinite(nll_hat)
            ok = false;
        end
    catch
        ok = false;
        k_hat = NaN;
        nll_hat = Inf;
    end
end

function [k_hat, beta_hat, eps_hat, nll_hat, ok] = safe_fit_model2(f, z0, epsMax)
% safe_fit_model2 Fit Model 2 in unconstrained space with parameter transforms:
%   k = exp(z1), beta = z2, eps = epsMax*sigmoid(z3)
    ok = true;
    try
        zhat = fminsearch(f, z0, optimset('Display','off'));
        k_hat    = exp(zhat(1));
        beta_hat = zhat(2);
        eps_hat  = epsMax * sigmoid(zhat(3));
        nll_hat  = f(zhat);
        if ~isfinite(k_hat) || ~isfinite(beta_hat) || ~isfinite(eps_hat) || ~isfinite(nll_hat)
            ok = false;
        end
    catch
        ok = false;
        k_hat = NaN; beta_hat = NaN; eps_hat = NaN; nll_hat = Inf;
    end
end

function p = clamp01(x)
% clamp01 Clip values element-wise to [0, 1].
    p = min(max(x, 0), 1);
end

function s = sigmoid(z)
% sigmoid Logistic sigmoid.
    s = 1 ./ (1 + exp(-z));
end

function z = logit(p)
% logit Inverse logistic with defensive probability clipping.
    p = min(max(p, 1e-12), 1-1e-12);
    z = log(p./(1-p));
end
