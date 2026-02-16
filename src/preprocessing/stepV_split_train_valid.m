function stepV_split_train_valid(timeMatPath, train_idx, outDir)
% STEPV_SPLIT_TRAIN_VALID  Split time-enriched participants into train/validation sets.
%
%   stepV_split_train_valid(timeMatPath, train_idx)
%   stepV_split_train_valid(timeMatPath, train_idx, outDir)
%
% This pipeline step creates explicit training and validation sets from the
% time-enriched participant structs generated in Step T1. The main goal is
% to support parameter identification on a subset of participants (training
% set), while reserving a separate subset for model validation.
%
% Typical usage:
%   % After running stepT1_add_times and producing
%   % derived/participants_time_stepT1.mat:
%   timeMatPath = "derived/participants_time_stepT1.mat";
%
%   % Suppose we want the first 24 participants for training:
%   train_idx = 1:24;
%
%   % Run the split step:
%   stepV_split_train_valid(timeMatPath, train_idx);
%
% This will produce:
%   - derived/participants_train_stepV.mat
%   - derived/participants_valid_stepV.mat
%
% Each file contains:
%   - participants_train / participants_valid (struct array)
%   - info struct with metadata:
%         .source_file
%         .train_indices
%         .valid_indices
%         .n_total
%         .n_train
%         .n_valid
%         .created
%
% Inputs:
%   timeMatPath  - Path to a MAT file containing time-enriched participants.
%                  This is typically the output of Step T1, for example:
%
%                      "derived/participants_time_stepT1.mat"
%
%                  The file is expected to contain either:
%
%                      participants_clean
%                  or  participants_with_time
%
%                  as an array of participant structs.
%
%   train_idx    - Vector of integer indices into the participants array
%                  defining the training set. The validation set will be
%                  the complement of train_idx in 1:N, where N is the
%                  total number of participants.
%
%   outDir       - Output directory for the resulting MAT files. If
%                  omitted or empty, defaults to "derived".
%
% Outputs:
%   This function does not return variables; instead, it writes the
%   following files to outDir:
%
%       outDir/participants_train_stepV.mat
%       outDir/participants_valid_stepV.mat
%
%   The saved variables are:
%       participants_train   - training participants (struct array)
%       participants_valid   - validation participants (struct array)
%       info                 - struct with metadata (same in both files)
%
% Assumptions:
%   - The participants in timeMatPath have already passed Steps 2–4 and T1,
%     meaning their structs contain:
%         P.doorTrials, P.trustProbes, P.questionnaires, P.reviews,
%         P.demographics, and time fields (e.g., .t_s) as required by
%         downstream steps (measurement mapping, trust simulation, etc.).
%   - No shuffling or randomization is performed here; train_idx is assumed
%     to be chosen externally (e.g., by script or fixed random seed).
%
% Example workflow:
%   1) Run preprocessing (Steps 1–4) and time enrichment (Step T1).
%   2) Decide on a split (e.g., 70% train, 30% validation).
%   3) Call stepV_split_train_valid to produce separate MAT files.
%   4) Use participants_train_stepV.mat for:
%        - measurement calibration (if desired),
%        - parameter fitting (trust_cost_all over training set).
%      Use participants_valid_stepV.mat for:
%        - out-of-sample evaluation and diagnostics.
%

    % ------------------------------------------------------------
    % 0) Handle inputs and defaults
    % ------------------------------------------------------------
    if nargin < 1 || isempty(timeMatPath)
        timeMatPath = "derived/participants_time_stepT1.mat";
    end

    if nargin < 2 || isempty(train_idx)
        error("stepV_split_train_valid:missing_train_idx", ...
              "You must provide 'train_idx' specifying training participant indices.");
    end

    if nargin < 3 || isempty(outDir)
        outDir = "derived";
    end

    % Ensure output directory exists
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    % ------------------------------------------------------------
    % 1) Load participants from time-enriched MAT file
    % ------------------------------------------------------------
    if ~isfile(timeMatPath)
        error("stepV_split_train_valid:file_not_found", ...
              "Time-enriched participants file not found: %s", timeMatPath);
    end

    S = load(timeMatPath);

    % Support both naming conventions for the participants array
    if isfield(S, "participants_clean")
        participants = S.participants_clean;
    elseif isfield(S, "participants_with_time")
        participants = S.participants_with_time;
    else
        error("stepV_split_train_valid:bad_file_contents", ...
              "File %s does not contain 'participants_clean' or 'participants_with_time'.", ...
              timeMatPath);
    end

    if ~isstruct(participants) || isempty(participants)
        error("stepV_split_train_valid:empty_participants", ...
              "No participants found in %s.", timeMatPath);
    end

    N = numel(participants);
    fprintf('[Step V] Loaded %d time-enriched participants from %s\n', N, timeMatPath);

    % ------------------------------------------------------------
    % 2) Split into training and validation sets
    % ------------------------------------------------------------
    [participants_train, participants_valid] = split_participants_by_index(participants, train_idx);

    n_train = numel(participants_train);
    n_valid = numel(participants_valid);

    % Derive validation indices as complement of train_idx
    all_idx = (1:N).';
    train_idx = unique(train_idx(:));           % ensure sorted, unique
    valid_idx = setdiff(all_idx, train_idx);

    fprintf('[Step V] Training set size   : %d participants\n', n_train);
    fprintf('[Step V] Validation set size : %d participants\n', n_valid);

    % ------------------------------------------------------------
    % 3) Build metadata struct
    % ------------------------------------------------------------
    info = struct();
    info.source_file   = timeMatPath;
    info.train_indices = train_idx;
    info.valid_indices = valid_idx;
    info.n_total       = N;
    info.n_train       = n_train;
    info.n_valid       = n_valid;
    info.created       = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    info.description   = [ ...
        "Train/validation split created from time-enriched participants " ...
        "for downstream model identification and validation." ...
    ];

    % ------------------------------------------------------------
    % 4) Save train and validation sets
    % ------------------------------------------------------------
    trainPath = fullfile(outDir, "participants_train_stepV.mat");
    validPath = fullfile(outDir, "participants_valid_stepV.mat");

    participants_train_out = participants_train; %#ok<NASGU>
    participants_valid_out = participants_valid; %#ok<NASGU>

    % Save training set
    participants_clean = participants_train_out; %#ok<NASGU>
    save(trainPath, "participants_clean", "info", "-v7.3");

    % Save validation set
    participants_clean = participants_valid_out; %#ok<NASGU>
    save(validPath, "participants_clean", "info", "-v7.3");

    fprintf('[Step V] Saved training participants to   %s\n', trainPath);
    fprintf('[Step V] Saved validation participants to %s\n', validPath);
end
