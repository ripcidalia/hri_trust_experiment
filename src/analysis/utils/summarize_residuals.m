function sumTbl = summarize_residuals(resTbl)
% summarize_residuals  Build overall + per-kind summary metrics.
%
% Returns one-row table with:
%   N_overall, wRMSE_overall, wMAE_overall, bias_overall, wBias_overall,
% plus embedded per-kind arrays for CSV friendliness.

    if isempty(resTbl) || height(resTbl) == 0
        sumTbl = table(0, NaN, NaN, NaN, NaN, {{}}, {[]}, {[]}, {[]}, {[]}, {[]}, ...
            'VariableNames', {'N_overall','wRMSE_overall','wMAE_overall','bias_overall','wBias_overall', ...
                              'kinds','N_by_kind','wRMSE_by_kind','wMAE_by_kind','bias_by_kind','wBias_by_kind'});
        return;
    end

    r = resTbl.residual;
    w = resTbl.weight;

    metO = compute_weighted_metrics(r, w);

    kinds = unique(resTbl.kind);
    N_kind     = NaN(numel(kinds),1);
    wRMSE_kind = NaN(numel(kinds),1);
    wMAE_kind  = NaN(numel(kinds),1);
    bias_kind  = NaN(numel(kinds),1);
    wBias_kind = NaN(numel(kinds),1);

    for i = 1:numel(kinds)
        mask = (resTbl.kind == kinds(i));
        metK = compute_weighted_metrics(resTbl.residual(mask), resTbl.weight(mask));
        N_kind(i)     = metK.N;
        wRMSE_kind(i) = metK.wRMSE;
        wMAE_kind(i)  = metK.wMAE;
        bias_kind(i)  = metK.bias;
        wBias_kind(i) = metK.wBias;
    end

    sumTbl = table();
    sumTbl.N_overall      = metO.N;
    sumTbl.wRMSE_overall  = metO.wRMSE;
    sumTbl.wMAE_overall   = metO.wMAE;
    sumTbl.bias_overall   = metO.bias;
    sumTbl.wBias_overall  = metO.wBias;

    sumTbl.kinds          = {cellstr(kinds)};
    sumTbl.N_by_kind      = {N_kind};
    sumTbl.wRMSE_by_kind  = {wRMSE_kind};
    sumTbl.wMAE_by_kind   = {wMAE_kind};
    sumTbl.bias_by_kind   = {bias_kind};
    sumTbl.wBias_by_kind  = {wBias_kind};
end
