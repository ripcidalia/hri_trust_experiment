function dstPath = safe_copy_into_dir(dstDir, srcPath, varargin)
% safe_copy_into_dir  Copy if exists, otherwise do nothing. Returns dstPath or "".

    srcPath = char(string(srcPath));
    if isfile(srcPath)
        dstPath = copy_into_dir(dstDir, srcPath, varargin{:});
    else
        dstPath = "";
    end
end
