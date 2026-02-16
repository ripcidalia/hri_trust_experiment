function out = compare_id_sets(idsA, idsB, labelA, labelB)
% compare_id_sets  Compare two participant ID sets (order-insensitive).
% Returns struct with ok flag + differences.

    if nargin < 3, labelA = "A"; end
    if nargin < 4, labelB = "B"; end

    idsA = string(idsA(:));
    idsB = string(idsB(:));

    out = struct();
    out.labelA = char(labelA);
    out.labelB = char(labelB);

    a = sort(unique(idsA));
    b = sort(unique(idsB));

    out.nA = numel(a);
    out.nB = numel(b);

    out.only_in_A = setdiff(a, b);
    out.only_in_B = setdiff(b, a);

    out.ok = isempty(out.only_in_A) && isempty(out.only_in_B);

    out.summary = sprintf("ID compare %s vs %s: nA=%d, nB=%d, ok=%d, onlyA=%d, onlyB=%d", ...
        labelA, labelB, out.nA, out.nB, out.ok, numel(out.only_in_A), numel(out.only_in_B));
end
