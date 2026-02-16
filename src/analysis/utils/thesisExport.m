function thesisExport(figHandle, outBase)
% thesisExport Export a figure to a vector PDF and a MATLAB .fig (for reproducibility).
%
% This helper provides a single, consistent export path for thesis figures.
% It saves:
%   - A vector PDF for inclusion in documents
%   - A .fig file for later inspection/editing in MATLAB
%
% INPUTS
%   figHandle (optional)
%       Figure handle to export. If omitted or empty, uses gcf.
%
%   outBase (string|char)
%       Output base path (without extension). The function writes:
%         <outBase>.pdf
%         <outBase>.fig
%
% OUTPUTS
%   (none)
%
% NOTES
%   - Uses exportgraphics when available for consistent vector PDF output.
%   - Falls back to print -dpdf -painters for older MATLAB versions.
%   - The figure is closed after saving to avoid accumulating hidden figures.
%

    if nargin < 1 || isempty(figHandle)
        figHandle = gcf;
    end
    if nargin < 2 || strlength(string(outBase)) == 0
        error("thesisExport requires an output base path (no extension).");
    end

    outBase = string(outBase);
    pdfPath = outBase + ".pdf";
    figPath = outBase + ".fig";

    % Ensure predictable export appearance.
    set(figHandle, 'Color', 'w');
    set(figHandle, 'InvertHardcopy', 'off');
    set(figHandle, 'PaperPositionMode', 'auto');

    % Prefer exportgraphics (modern, consistent, vector output).
    try
        exportgraphics(figHandle, pdfPath, ...
            'ContentType', 'vector', ...
            'BackgroundColor', 'white');
    catch
        % Fallback for older MATLAB versions.
        set(figHandle, 'PaperPositionMode', 'auto');
        print(figHandle, pdfPath, '-dpdf', '-painters');
    end

    % Save MATLAB figure for reproducibility (useful during thesis iteration).
    savefig(figHandle, figPath);

    % Close to avoid accumulating off-screen figures in batch runs.
    close(figHandle);
end
