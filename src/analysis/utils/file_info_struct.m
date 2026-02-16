function s = file_info_struct(path)
% file_info_struct  Lightweight provenance info for a file (no hashing).
    path = char(string(path));
    s = struct();
    s.path = path;
    if isfile(path)
        d = dir(path);
        s.exists   = true;
        s.bytes    = d.bytes;
        s.datenum  = d.datenum;
        try
            s.modified = char(datetime(d.datenum, 'ConvertFrom','datenum', 'Format','yyyy-MM-dd HH:mm:ss'));
        catch
            s.modified = '';
        end
    else
        s.exists   = false;
        s.bytes    = NaN;
        s.datenum  = NaN;
        s.modified = '';
    end
end
