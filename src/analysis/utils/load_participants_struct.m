function participants = load_participants_struct(matPath)
% load_participants_struct  Load participant struct array from a MAT file.
% Supports common variable names used across pipeline artefacts.

    S = load(matPath);

    if isfield(S, "participants_probes_mapped")
        participants = S.participants_probes_mapped;
        return;
    end
    if isfield(S, "participants_mapped")
        participants = S.participants_mapped;
        return;
    end
    if isfield(S, "participants_clean")
        participants = S.participants_clean;
        return;
    end

    % Fallback: single struct array variable
    fn = fieldnames(S);
    for k = 1:numel(fn)
        v = S.(fn{k});
        if isstruct(v) && ~isempty(v)
            participants = v;
            return;
        end
    end

    error("Could not find participants struct array in %s.", matPath);
end
