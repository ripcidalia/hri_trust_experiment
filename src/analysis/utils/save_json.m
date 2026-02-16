function save_json(path, S)
% save_json  Save struct as pretty JSON (best-effort).
    path = char(string(path));
    try
        txt = jsonencode(S);
        % Pretty-print best-effort: insert newlines after commas/braces.
        txt = regexprep(txt, ',"', sprintf(',\n  "'));
        txt = regexprep(txt, '{', sprintf('{\n  '));
        txt = regexprep(txt, '}', sprintf('\n}\n'));
        fid = fopen(path, 'w');
        if fid < 0
            warning("Could not open %s for writing JSON.", string(path));
            return;
        end
        fwrite(fid, txt, 'char');
        fclose(fid);
    catch ME
        warning("Failed to write JSON manifest: %s", ME.message);
    end
end
