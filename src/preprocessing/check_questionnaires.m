function [ok, msg] = check_questionnaires(P)
% check_questionnaires  Validate presence and basic integrity of questionnaires.
%
%   [ok, msg] = check_questionnaires(P)
%
% This function performs structural and value-level checks on the
% questionnaire data stored in P.questionnaires. It enforces that:
%
%   - All four questionnaires are present:
%       - t40_pre,  t40_post  (40-item trust scale)
%       - t14_mid1, t14_mid2  (14-item mid-block trust scale)
%
%   - For the 14-item questionnaires:
%       - Field total_percent exists and is coercible to a finite scalar.
%
%   - For the 40-item questionnaires:
%       - Fields total_percent and trust14_equiv_total_percent exist.
%       - Both fields are coercible to finite scalar values.
%
% Inputs:
%   P   - Participant struct with field:
%           P.questionnaires : struct with fields t40_pre, t40_post,
%                              t14_mid1, t14_mid2 (if present).
%
% Outputs:
%   ok  - Logical flag indicating whether all questionnaire checks pass.
%   msg - Diagnostic message describing the first issue encountered
%         (empty if ok == true).

    ok  = true;
    msg = "";

    % ---------------------------------------------------------------------
    % 1) Presence of questionnaires struct
    % ---------------------------------------------------------------------
    if ~isfield(P, "questionnaires") || isempty(P.questionnaires)
        ok  = false;
        msg = "No questionnaires present.";
        return;
    end
    Q = P.questionnaires;

    % ---------------------------------------------------------------------
    % 2) Require presence of all four questionnaire entries
    % ---------------------------------------------------------------------
    requiredNames = ["t40_pre","t40_post","t14_mid1","t14_mid2"];
    for nm = requiredNames
        if ~isfield(Q, nm) || isempty(Q.(nm))
            ok  = false;
            msg = sprintf("Questionnaire '%s' missing.", nm);
            return;
        end
    end

    % ---------------------------------------------------------------------
    % 3) 14-item questionnaires (t14_mid1, t14_mid2)
    %     - require total_percent, coercible to finite scalar
    % ---------------------------------------------------------------------
    for nm = ["t14_mid1","t14_mid2"]
        q = Q.(nm);

        % total_percent must exist
        if ~isfield(q, "total_percent")
            ok = false;
            msg = sprintf("Questionnaire '%s' missing total_percent.", nm);
            return;
        end

        % Coerce total_percent to scalar double from any type
        val = q.total_percent;
        if iscell(val)
            if ~isempty(val)
                val = val{1};
            else
                ok  = false;
                msg = sprintf("Questionnaire '%s' total_percent empty cell.", nm);
                return;
            end
        end
        if isstring(val) || ischar(val)
            % Strip non-numeric characters and convert
            val = str2double(regexprep(strtrim(string(val)), '[^0-9\.\-eE]+', ''));
        end
        if ~(isnumeric(val) && isscalar(val) && ~isnan(val))
            ok  = false;
            msg = sprintf("Questionnaire '%s' total_percent not a finite scalar.", nm);
            return;
        end
    end

    % ---------------------------------------------------------------------
    % 4) 40-item questionnaires (t40_pre, t40_post)
    %     - require both total_percent and trust14_equiv_total_percent,
    %       each coercible to finite scalar.
    % ---------------------------------------------------------------------
    for nm = ["t40_pre","t40_post"]
        q = Q.(nm);

        % Both fields must be present
        if ~isfield(q, "total_percent")
            ok  = false;
            msg = sprintf("Questionnaire '%s' missing 40-item total_percent.", nm);
            return;
        end
        if ~isfield(q, "trust14_equiv_total_percent")
            ok  = false;
            msg = sprintf("Questionnaire '%s' missing trust14_equiv_total_percent.", nm);
            return;
        end

        % --- total_percent ---
        val1 = q.total_percent;
        if iscell(val1)
            if ~isempty(val1)
                val1 = val1{1};
            else
                ok  = false;
                msg = sprintf("Questionnaire '%s' 40-item total_percent empty cell.", nm);
                return;
            end
        end
        if isstring(val1) || ischar(val1)
            val1 = str2double(regexprep(strtrim(string(val1)), '[^0-9\.\-eE]+', ''));
        end
        if ~(isnumeric(val1) && isscalar(val1) && ~isnan(val1))
            ok  = false;
            msg = sprintf("Questionnaire '%s' 40-item total_percent not a finite scalar.", nm);
            return;
        end

        % --- trust14_equiv_total_percent ---
        val2 = q.trust14_equiv_total_percent;
        if iscell(val2)
            if ~isempty(val2)
                val2 = val2{1};
            else
                ok  = false;
                msg = sprintf("Questionnaire '%s' trust14_equiv_total_percent empty cell.", nm);
                return;
            end
        end
        if isstring(val2) || ischar(val2)
            val2 = str2double(regexprep(strtrim(string(val2)), '[^0-9\.\-eE]+', ''));
        end
        if ~(isnumeric(val2) && isscalar(val2) && ~isnan(val2))
            ok  = false;
            msg = sprintf("Questionnaire '%s' trust14_equiv_total_percent not a finite scalar.", nm);
            return;
        end
    end
end
