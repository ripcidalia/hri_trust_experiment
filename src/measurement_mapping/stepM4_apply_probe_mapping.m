function stepM4_apply_probe_mapping(cleanMatPath, calibPath, outPath)
% stepM4_apply_probe_mapping  Populate 40-scale equivalents for all trust probes.
%
%   stepM4_apply_probe_mapping(cleanMatPath, calibPath, outPath)
%
% This Step M4 function augments every trust probe with a 40-item–scale
% equivalent value. For probes associated with questionnaire time points,
% the 40-scale value is taken directly from the corresponding questionnaire
% total (or mapped total) to enforce consistency at those anchors. For all
% other probes, a global linear probe→40 mapping is applied.
%
% For each probe entry U(k):
%   - If questionnaire_type is one of:
%         "t40_pre", "t40_post", "t14_mid1", "t14_mid2"
%     then:
%         value_40 is set to the corresponding questionnaire total on the
%         40-item scale, and t_s is copied from that questionnaire entry.
%
%   - Otherwise, if the probe has a valid numeric "value", then:
%         value_40 = a1 * value + b1
%
%   - If the probe value is NaN, value_40 is set to NaN.
%
% Inputs:
%   cleanMatPath : (optional) Path to MAT file containing 'participants_mapped'
%                 (typically output of Step M2). Default:
%                     "derived/participants_mapped14_stepM2.mat"
%
%   calibPath    : (optional) Path to MAT file containing 'calib' with fields:
%                     calib.a1
%                     calib.b1
%                 Default:
%                     "derived/measurement_stepM1_calibration.mat"
%
%   outPath      : (optional) Output MAT file path.
%                 Default: "derived/participants_probes_mapped_stepM4.mat".
%
% Output (file):
%   Writes MAT file specified by outPath containing:
%       participants_probes_mapped : participants_mapped with each probe struct
%           extended by field:
%               .value_40  (probe expressed on 40-item scale)
%           and, for anchor-typed probes, field:
%               .t_s       (timestamp copied from the questionnaire entry)
%       info : struct with metadata (source paths, mapping coefficients, timestamps).
%
% Assumptions:
%   - participants_mapped(i).trustProbes is an array of probe structs.
%   - Each probe struct has fields "value" and "questionnaire_type".
%   - Questionnaire structs contain the relevant totals and timestamps:
%         t40_pre.total_percent,   t40_pre.t_s
%         t40_post.total_percent,  t40_post.t_s
%         t14_mid1.total_percent_40, t14_mid1.t_s
%         t14_mid2.total_percent_40, t14_mid2.t_s
%   - No existing structure or field names are modified; fields value_40 (and
%     t_s for anchor-typed probes) are added/overwritten on the probe entries.

    if nargin < 1 || isempty(cleanMatPath)
        cleanMatPath = "derived/participants_mapped14_stepM2.mat";
    end
    if nargin < 2 || isempty(calibPath)
        calibPath = "derived/measurement_stepM1_calibration.mat";
    end
    if nargin < 3 || isempty(outPath)
        outPath = "derived/participants_probes_mapped_stepM4.mat";
    end

    if ~isfile(cleanMatPath)
        error("participants_mapped file not found: %s", cleanMatPath);
    end
    if ~isfile(calibPath)
        error("calib file not found: %s", calibPath);
    end

    % ------------------------------------------------------------
    % 1) Load participants and probe mapping coefficients
    % ------------------------------------------------------------
    S = load(cleanMatPath, "participants_mapped");
    participants = S.participants_mapped;

    C = load(calibPath, "calib");
    calib = C.calib;
    a1 = calib.a1;
    b1 = calib.b1;
    
    % ------------------------------------------------------------
    % 2) Populate 40-scale values for every probe entry
    % ------------------------------------------------------------
    N = numel(participants);
    for i = 1:N
        U = participants(i).trustProbes;
        for k = 1:numel(U)
            val  = U(k).value;
            type = string(U(k).questionnaire_type);

            if ~isnan(val)
                % For probes tied to questionnaire completion, use the
                % questionnaire-derived 40-scale totals and timestamps.
                if type == "t40_pre"
                    participants(i).trustProbes(k).value_40 = participants(i).questionnaires.t40_pre.total_percent;
                    participants(i).trustProbes(k).t_s     = participants(i).questionnaires.t40_pre.t_s;                    
                elseif type == "t40_post"
                    participants(i).trustProbes(k).value_40 = participants(i).questionnaires.t40_post.total_percent;
                    participants(i).trustProbes(k).t_s     = participants(i).questionnaires.t40_post.t_s;
                elseif type == "t14_mid1"
                    participants(i).trustProbes(k).value_40 = participants(i).questionnaires.t14_mid1.total_percent_40;
                    participants(i).trustProbes(k).t_s     = participants(i).questionnaires.t14_mid1.t_s;
                elseif type == "t14_mid2"
                    participants(i).trustProbes(k).value_40 = participants(i).questionnaires.t14_mid2.total_percent_40;
                    participants(i).trustProbes(k).t_s     = participants(i).questionnaires.t14_mid2.t_s;
                else
                    % For all other probes, apply the global linear mapping
                    participants(i).trustProbes(k).value_40 = a1 * double(val) + b1;
                end
            else
                % Preserve NaN for missing or invalid probe values
                participants(i).trustProbes(k).value_40 = NaN;
            end        
        end
    end

    % ------------------------------------------------------------
    % 3) Save updated participants with mapped probes
    % ------------------------------------------------------------
    if ~isfolder(fileparts(outPath))
        mkdir(fileparts(outPath));
    end

    participants_probes_mapped = participants;

    info = struct();
    info.source_mapped_file = cleanMatPath;
    info.calib_file         = calibPath;
    info.a1                 = a1;
    info.b1                 = b1;
    info.created            = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    info.n_participants     = N;

    save(outPath, "participants_probes_mapped", "info", "-v7.3");

    fprintf('[Step M4] Applied probe→40 mapping to all probes for %d participants.\n', N);
    fprintf('          Saved to %s\n', outPath);
end
