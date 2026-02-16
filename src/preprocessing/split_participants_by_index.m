function [participants_train, participants_valid] = split_participants_by_index(participants, train_idx)
% split_participants_by_index  Split participants into train / validation sets.
%
%   [participants_train, participants_valid] = ...
%       split_participants_by_index(participants, train_idx)
%
% This utility function splits an array of participant structs into two
% disjoint subsets:
%
%   - A training set, indexed by train_idx.
%   - A validation set, containing all remaining participants.
%
% It does not perform any randomization; it simply uses the indices
% provided by the caller. This is intended to be used after the main
% preprocessing and time-enrichment steps, when each participant struct
% already contains:
%
%   P.timeline, P.doorTrials, P.trustProbes, P.questionnaires, P.reviews,
%   P.demographics, and any time fields (e.g., .t_s) required by the
%   trust model.
%
% Inputs:
%   participants  - array of participant structs (1xN or Nx1), typically
%                   loaded from participants_time_stepT1.mat.
%
%   train_idx     - vector of integer indices into 'participants' defining
%                   the training set (e.g., [1 2 5 7]). The validation set
%                   is implicitly defined as the complement:
%
%                       valid_idx = setdiff(1:N, train_idx)
%
% Outputs:
%   participants_train  - array of participant structs corresponding to
%                         participants(train_idx).
%
%   participants_valid  - array of participant structs corresponding to
%                         participants(valid_idx), where valid_idx is the
%                         complement of train_idx in 1:numel(participants).
%
% Assumptions and notes:
%   - train_idx does not need to be sorted; any order is allowed.
%   - Duplicate indices in train_idx are ignored (they are de-duplicated).
%   - Out-of-range indices (<=0 or >N) will trigger an error.
%   - If train_idx covers all participants, the validation set will be
%     empty; if train_idx is empty, the training set will be empty.
%
% Example:
%   % Suppose there are 20 participants:
%   S = load("derived/participants_time_stepT1.mat", "participants_clean");
%   P = S.participants_clean;
%
%   % Choose first 15 as training:
%   train_idx = 1:15;
%   [P_train, P_valid] = split_participants_by_index(P, train_idx);

    % Ensure participants is a non-empty struct array
    if ~isstruct(participants) || isempty(participants)
        error("split_participants_by_index:invalid_participants", ...
              "Input 'participants' must be a non-empty struct array.");
    end

    % Number of participants
    N = numel(participants);

    % Basic checks on train_idx
    if nargin < 2 || isempty(train_idx)
        % Empty training set is technically allowed, but warn to avoid
        % silent mistakes.
        warning("split_participants_by_index:empty_train_idx", ...
                "train_idx is empty; training set will be empty and all participants go to validation.");
        train_idx = [];
    end

    % Force column vector of unique indices
    train_idx = unique(train_idx(:));

    % Check that indices are integers and within [1, N]
    if any(train_idx < 1) || any(train_idx > N) || any(train_idx ~= round(train_idx))
        error("split_participants_by_index:invalid_indices", ...
              "train_idx must contain valid integer indices in the range [1, %d].", N);
    end

    % Logical mask for training participants
    train_mask = false(N,1);
    train_mask(train_idx) = true;

    % Complement mask for validation participants
    valid_mask = ~train_mask;

    % Perform the actual split
    participants_train = participants(train_mask);
    participants_valid = participants(valid_mask);
end
