function ensure_dir(dirPath)
% ensure_dir  Create directory if it doesn't exist.
    dirPath = char(string(dirPath));
    if ~isfolder(dirPath)
        mkdir(dirPath);
    end
end
