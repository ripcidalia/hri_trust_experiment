function reviews = extract_reviews(Tk)
% extract_reviews  Extract reputation/review phase data for one participant.
%
%   reviews = extract_reviews(Tk)
%
% This function collects all events associated with the reputation / online
% reviews phase from a participant-level event table Tk. It identifies rows
% where event_type == "reputation_item" and stores a compact description of
% those items, together with overall review-condition metadata if present.
%
% The returned struct has the fields:
%   reviews.items            : 1×N struct array, one struct per reputation
%                              item event, containing selected raw columns.
%   reviews.review_condition : descriptive label of the review condition
%                              (e.g., "very_positive"), if available.
%   reviews.review_expected  : numeric summary of the expected reputation
%                              condition, if available.
%
% For each reputation item, the following fields may be present in
% reviews.items(i), depending on the columns available in Tk:
%   review_ids
%   review_tones
%   review_avatars
%   response              : raw response string (if available)
%   response_struct       : decoded JSON struct (if available)
%
% Inputs:
%   Tk - Table containing all events for a single participant, including
%        at least 'event_type' and, for review items, columns such as
%        review_ids, review_tones, review_avatars, response,
%        response_struct, review_condition, review_expected.
%
% Outputs:
%   reviews - Struct summarizing the reputation phase for this participant.
%             If no reputation_item events are found, reviews.items is []
%             and review_condition / review_expected remain defaulted.

    % ---------------------------------------------------------------------
    % 1) Select rows corresponding to reputation items
    % ---------------------------------------------------------------------
    mask = (string(Tk.event_type) == "reputation_item");
    R = Tk(mask, :);

    % Initialize output struct with defaults.
    reviews = struct( ...
        'items', [], ...
        'review_condition', "", ...
        'review_expected', NaN);

    % If there are no reputation items, return defaults.
    if isempty(R)
        return;
    end

    % ---------------------------------------------------------------------
    % 2) Copy selected columns for each reputation item
    % ---------------------------------------------------------------------
    % Keep only a subset of columns that are useful for modelling or
    % inspection, and that are expected to exist in the raw table.
    keepCols = intersect( ...
        ["review_ids","review_tones","review_avatars","response","response_struct"], ...
        string(R.Properties.VariableNames));

    items = struct([]);
    for i = 1:height(R)
        item = struct();
        for c = 1:numel(keepCols)
            name = keepCols(c);
            val = R.(name)(i);
            if name == "response_struct"
                % Keep JSON-decoded form as-is (struct or other type).
                item.response_struct = val;
            else
                % Store other kept columns as string for consistency.
                item.(name) = string(val);
            end
        end
        items = [items; item]; %#ok<AGROW>  % grow array; small N expected
    end
    reviews.items = items;

    % ---------------------------------------------------------------------
    % 3) Extract condition and expected score (first non-missing values)
    % ---------------------------------------------------------------------
    reviews.review_condition = get_string_field(R, "review_condition");
    reviews.review_expected  = get_numeric_field(R, "review_expected");
end
