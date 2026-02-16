function dstPath = copy_into_dir(dstDir, srcPath, varargin)
% copy_into_dir  Copy a file into a directory. Returns destination path.
%
% Name-value:
%   "Overwrite" (false)  - allow overwrite
%   "RenameTo"  ("")     - optional new filename (with extension)

    p = inputParser;
    p.addRequired("dstDir",  @(s) isstring(s) || ischar(s));
    p.addRequired("srcPath", @(s) isstring(s) || ischar(s));
    p.addParameter("Overwrite", false, @(x) islogical(x) && isscalar(x));
    p.addParameter("RenameTo", "", @(s) isstring(s) || ischar(s));
    p.parse(dstDir, srcPath, varargin{:});
    args = p.Results;

    dstDir  = char(string(args.dstDir));
    srcPath = char(string(args.srcPath));
    ensure_dir(dstDir);

    [~, name, ext] = fileparts(srcPath);
    renameTo = string(args.RenameTo);
    if strlength(renameTo) > 0
        dstFile = char(renameTo);
    else
        dstFile = [name ext];
    end

    dstPath = fullfile(dstDir, dstFile);

    if isfile(dstPath) && ~args.Overwrite
        error("Refusing to overwrite existing file: %s", string(dstPath));
    end

    ok = copyfile(srcPath, dstPath);
    if ~ok
        error("Failed to copy %s -> %s", string(srcPath), string(dstPath));
    end
end
