function save_participant_metric_figure(partTbl, outDir, outBaseName, S)
% save_participant_metric_figure  Bar plot of per-participant wRMSE.

    if isempty(partTbl) || height(partTbl) == 0
        return;
    end

    f = figure('Visible','off','Color','w','Name','Participant wRMSE');
    thesisStyle(f);

    b = bar(partTbl.wRMSE);
    b.EdgeColor = 'none';

    grid on;
    xlim([0 height(partTbl)+1]);
    xlabel('Participant index', 'FontSize', S.fontSizeLbl);
    ylabel('wRMSE', 'FontSize', S.fontSizeYlb);
    title('Per-participant wRMSE', 'FontSize', S.fontSizeTit);

    local_save_figure(f, fullfile(outDir, outBaseName), S);
end

function local_save_figure(f, outBase, S)
% local_save_figure Finalize and export a figure using thesis styling helpers.
    thesisFinalizeFigure(f, S);
    thesisExport(f, outBase);
end
