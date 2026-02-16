% ExperimentalDataProcessing/preprocessing.m
%
% PREPROCESSING  Run the full preprocessing pipeline for the SAR trust study.
%
% This script orchestrates the preprocessing and data structuring steps for
% the human–robot trust experiment. It:
%   1) Loads and normalizes raw experimental event logs from CSV (Step 1).
%   2) Builds per-participant structures (Step 2).
%   3) Performs validation of participant timelines and events (Step 3).
%   4) Applies inclusion/exclusion criteria to select valid participants (Step 4).
%   5) Adds continuous-time information to the cleaned participants (Step T1).
%
% The individual steps are implemented in dedicated functions under src/:
%   - step1_loader.m              : Normalize raw events and save derived/normalized_events_step1.mat
%   - step2_build_participants.m  : Build participant structs from normalized events
%   - step3_validate.m            : Run consistency checks and save validation reports
%   - step4_filter_participants.m : Apply participant filters and save cleaned set
%   - stepT1_add_times.m          : Construct time grids and add timing information
%
% This script is intended to be run from the project root and assumes:
%   - Raw event data is available at  rawData/events.csv
%   - Derived output is written under the derived/ directory as used below.
%
% No functionality is implemented here; this file only sequences the calls
% to the preprocessing functions. Do not modify the function calls or file
% paths without updating the rest of the pipeline accordingly.

%% Step 1: Load and normalize raw events
clear; clc;
addpath(genpath("src"));

csvPath = "rawData/events.csv";
step1_loader(csvPath);

%% Step 2: Build participant structures
step2_build_participants("derived/normalized_events_step1.mat");

%% Step 3: Validate participants and timelines
step3_validate("derived/participants_step2.mat");

%% Step 4: Apply inclusion/exclusion filters

% Example filter options (see step4_filter_participants.m for full option list):
%
% opts = struct();
% opts.gender            = "woman";   % Filter by reported gender
% opts.set_id            = "SetA";    % Filter by experimental set identifier
% opts.set_id_mode       = "exclude"; % "include" or "exclude" this set_id
% opts.device_type       = "mobile";  % Filter by device type
% opts.device_type_mode  = "exclude"; % "include" or "exclude" this device_type
%
% To use custom filters, pass opts as the third argument to step4_filter_participants.

step4_filter_participants("derived/participants_step2.mat", ...
                          "derived/validation_report_step3.mat");

%% Step T1: Add timing information to cleaned participants
stepT1_add_times("derived/participants_clean_step4.mat", ...
                 "rawData/events.csv");

%% Step V: Create training/validation split on participants
S = load("derived/participants_clean_step4.mat", "participants_clean");
train_idx = 1:length(S.participants_clean); % initialize training idx vector
clear S;
valid_idx = [1, 2, 3, 5, 6, 9, 21, 22, 23, 32]; % choose participant indexes for model validation 
train_idx = train_idx(setdiff(train_idx,valid_idx)); % remove validation participants from training set

stepV_split_train_valid("derived/participants_time_stepT1.mat", ...
                 train_idx);
