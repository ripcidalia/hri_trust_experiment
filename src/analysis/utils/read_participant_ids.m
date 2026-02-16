function ids = read_participant_ids(P)
% read_participant_ids  Extract participant_id as string array.
    if isempty(P)
        ids = string.empty(0,1);
        return;
    end
    if ~isfield(P, "participant_id")
        error("Participant struct missing field 'participant_id'.");
    end
    ids = string({P.participant_id}');
end
