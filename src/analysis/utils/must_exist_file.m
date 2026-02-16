function must_exist_file(path, label)
% must_exist_file  Error if file does not exist.
    if nargin < 2, label = "File"; end
    path = char(string(path));
    if ~isfile(path)
        error("%s not found: %s", string(label), string(path));
    end
end
