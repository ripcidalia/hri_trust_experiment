function stepM2_apply_14mapping(cleanMatPath, calibPath, outPath)
% stepM2_apply_14mapping  Apply 14→40 calibration to mid-block questionnaire scores.
%
%   stepM2_apply_14mapping(cleanMatPath, calibPath, outPath)
%
% This Step M2 script loads cleaned participant data and the global
% calibration coefficients produced in Step M1, then maps mid-block 14-item
% questionnaire totals onto the 40-item percentage scale. The mapped values
% are stored alongside the original mid-block totals in each participant's
% questionnaires struct.
%
% Specifically, for each participant:
%   - If t14_mid1.total_percent exists, compute:
%         t14_mid1.total_percent_40 = a14 * t14_mid1.total_percent + b14
%   - If t14_mid2.total_percent exists, compute:
%         t14_mid2.total_percent_40 = a14 * t14_mid2.total_percent + b14
% Missing or non-coercible values are stored as NaN, and output fields are
% created for consistency across participants.
%
% Inputs:
%   cleanMatPath : (optional) Path to MAT file containing 'participants_clean'.
%                  Default: "derived/participants_time_stepT1.mat".
%
%   calibPath    : (optional) Path to MAT file containing 'calib' with fields:
%                      calib.a14
%                      calib.b14
%                  Default: "derived/measurement_stepM1_calibration.mat".
%
%   outPath      : (optional) Output MAT file path.
%                  Default: "derived/participants_mapped14_stepM2.mat".
%
% Output (file):
%   Writes MAT file specified by outPath containing:
%
%     participants_mapped : struct array; a copy of participants_clean with
%                           added fields:
%                               P(i).questionnaires.t14_mid1.total_percent_40
%                               P(i).questionnaires.t14_mid2.total_percent_40
%
%     info                : metadata struct with fields:
%                               .source_clean_file
%                               .calib_file
%                               .created
%                               .n_participants
%
% Assumptions:
%   - P(i).questionnaires.t14_mid1.total_percent and/or t14_mid2.total_percent
%     may be missing or empty for participants who did not complete them.
%   - 14-item totals are expressed in percent units (0..100) compatible with
%     the calibration scale.

    if nargin < 1 || isempty(cleanMatPath)
        cleanMatPath = "derived/participants_time_stepT1.mat";
    end
    if nargin < 2 || isempty(calibPath)
        calibPath = "derived/measurement_stepM1_calibration.mat";
    end
    if nargin < 3 || isempty(outPath)
        outPath = "derived/participants_mapped14_stepM2.mat";
    end

    if ~isfile(cleanMatPath)
        error("Clean participants file not found: %s", cleanMatPath);
    end
    if ~isfile(calibPath)
        error("Calibration file not found: %s", calibPath);
    end

    % ------------------------------------------------------------
    % 1) Load cleaned participants
    % ------------------------------------------------------------
    S = load(cleanMatPath, "participants_clean");
    if ~isfield(S, "participants_clean")
        error("File %s does not contain 'participants_clean'.", cleanMatPath);
    end
    participants = S.participants_clean;
    N = numel(participants);

    % ------------------------------------------------------------
    % 2) Load 14→40 calibration coefficients (a14, b14)
    % ------------------------------------------------------------
    C = load(calibPath, "calib");
    if ~isfield(C, "calib")
        error("File %s does not contain 'calib'.", calibPath);
    end
    calib = C.calib;

    a14 = calib.a14;
    b14 = calib.b14;

    fprintf('[Step M2] Using 14→40 mapping: Q40 = %.4f * Q14 + %.4f\n', a14, b14);

    % ------------------------------------------------------------
    % 3) Apply linear mapping to mid-block 14-item totals
    % ------------------------------------------------------------
    for i = 1:N
        Q = participants(i).questionnaires;

        % --- t14_mid1 ---
        if isfield(Q, "t14_mid1") && ~isempty(Q.t14_mid1) ...
                && isfield(Q.t14_mid1, "total_percent")

            % Coerce stored value to a numeric scalar when possible
            val = Q.t14_mid1.total_percent;
            val_num = coerce_scalar_double(val);

            % Apply calibration if a valid numeric scalar is available
            if ~isnan(val_num)
                mapped = a14 * val_num + b14;
            else
                mapped = NaN;
            end

            participants(i).questionnaires.t14_mid1.total_percent_40 = mapped;
        else
            % Ensure field exists for consistent downstream access
            participants(i).questionnaires.t14_mid1.total_percent_40 = NaN;
        end

        % --- t14_mid2 ---
        if isfield(Q, "t14_mid2") && ~isempty(Q.t14_mid2) ...
                && isfield(Q.t14_mid2, "total_percent")

            val = Q.t14_mid2.total_percent;
            val_num = coerce_scalar_double(val);

            if ~isnan(val_num)
                mapped = a14 * val_num + b14;
            else
                mapped = NaN;
            end

            participants(i).questionnaires.t14_mid2.total_percent_40 = mapped;
        else
            participants(i).questionnaires.t14_mid2.total_percent_40 = NaN;
        end
    end

    % ------------------------------------------------------------
    % 4) Save updated participants with mapped mid-block scores
    % ------------------------------------------------------------
    if ~isfolder(fileparts(outPath))
        mkdir(fileparts(outPath));
    end

    participants_mapped = participants;
    info = struct();
    info.source_clean_file = cleanMatPath;
    info.calib_file        = calibPath;
    info.created           = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    info.n_participants    = N;

    save(outPath, "participants_mapped", "info", "-v7.3");

    fprintf('[Step M2] Applied 14→40 mapping to mid-block questionnaires for %d participants.\n', N);
    fprintf('          Saved to %s\n', outPath);
end

% -------------------------------------------------------------------------
% Local helper: coerce_scalar_double
% -------------------------------------------------------------------------
function v = coerce_scalar_double(val)
% COERCE_SCALAR_DOUBLE  Convert a stored value to a scalar double when possible.
%
%   v = coerce_scalar_double(val)
%
% Attempts to coerce "val" into a scalar double:
%   - cell: unwraps the first element and retries
%   - numeric: returns double(val) if scalar; otherwise returns NaN
%   - string/char: trims, removes non-numeric characters, and uses str2double
%
% Returns NaN if conversion is not possible or if the result is not scalar.

    v = NaN;

    if iscell(val)
        if isempty(val), return; end
        val = val{1};
    end

    if isnumeric(val)
        if isscalar(val)
            v = double(val);
        end
        return;
    end

    if isstring(val) || ischar(val)
        s = regexprep(strtrim(string(val)), '[^0-9\.\-eE]+', '');
        vv = str2double(s);
        if isscalar(vv)
            v = vv;
        end
        return;
    end
end
