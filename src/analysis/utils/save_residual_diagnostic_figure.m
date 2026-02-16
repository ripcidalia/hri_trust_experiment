function save_residual_diagnostic_figure(resTbl, outDir, outBaseName, S)
% save_residual_diagnostic_figure  2x2 residual diagnostics figure.

    if isempty(resTbl) || height(resTbl) == 0
        return;
    end

    r    = resTbl.residual;
    yhat = resTbl.y_hat;
    t    = resTbl.t_s;
    kind = resTbl.kind;

    f = figure('Visible','off','Color','w','Name','Residual diagnostics');
    thesisStyle(f);

    set(f,'Units','centimeters');
    set(f,'Position',[2 2 S.figSizeTrajectoryGrid]);

    tiledlayout(2,2,'Padding','loose','TileSpacing','loose');

    % 1) Histogram
    nexttile;
    histogram(r, 30);
    grid on;
    xlabel('Residual ($\widehat{q} - \widehat{\tau}$)', 'FontSize', S.fontSizeLbl);
    ylabel('Count', 'FontSize', S.fontSizeYlb);
    title('Residual histogram', 'FontSize', S.fontSizeTit);

    % 2) Boxplot by kind (robust ordering)
    nexttile;
    for i=1:length(kind)
        kind(i) = measurementDisplayName(kind(i));
    end
    desired = ["$q^{(1)}$","$q^{(14)}_{\mathrm{mid1}}$","$q^{(14)}_{\mathrm{mid2}}$","$q^{(40)}_{\mathrm{post}}$"];
    kindCat = categorical(kind, desired, 'Ordinal', true);

    keep = ~isundefined(kindCat);
    boxplot(r(keep), kindCat(keep));
    grid on;
    xlabel('Measurement type', 'FontSize', S.fontSizeLbl);
    ylabel('Residual', 'FontSize', S.fontSizeYlb);
    title('Residuals by measurement type', 'FontSize', S.fontSizeTit);

    % 3) Residual vs predicted
    nexttile;
    plot(yhat, r, '.', 'MarkerSize', 8);
    grid on;
    xlabel('$\widehat{\tau}$', 'FontSize', S.fontSizeLbl);
    ylabel('Residual', 'FontSize', S.fontSizeYlb);
    title('Residual vs predicted', 'FontSize', S.fontSizeTit);

    % 4) Residual vs time
    nexttile;
    plot(t, r, '.', 'MarkerSize', 8);
    grid on;
    xlabel('Time [s]', 'FontSize', S.fontSizeLbl);
    ylabel('Residual', 'FontSize', S.fontSizeYlb);
    title('Residual vs time', 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end