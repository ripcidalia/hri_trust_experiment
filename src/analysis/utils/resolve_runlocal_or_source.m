function resolved = resolve_runlocal_or_source(archiveDir, filesStruct, key, preferredBasename, label)
    archiveDir = string(archiveDir);

    % 1) Prefer the run-local copy in archive_dir (A1 copies these by basename)
    candidate = fullfile(archiveDir, preferredBasename);
    if isfile(candidate)
        resolved = string(candidate);
        return;
    end

    % 2) Otherwise fall back to manifest file_info_struct path
    if ~isfield(filesStruct, key)
        error("A1 manifest missing runInfo.files.%s (needed for %s).", key, label);
    end
    fi = filesStruct.(key);

    if isstruct(fi) && isfield(fi, "path") && ~isempty(fi.path) && isfile(fi.path)
        resolved = string(fi.path);
        return;
    end

    % 3) Last resort: if key is already a string path (older manifests)
    if ischar(fi) || isstring(fi)
        if isfile(fi)
            resolved = string(fi);
            return;
        end
    end

    error("Could not resolve %s. Tried run-local %s and manifest %s.path.", label, candidate, key);
end