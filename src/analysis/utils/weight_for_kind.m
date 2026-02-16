function w = weight_for_kind(kindStr, weights)
% weight_for_kind  Map measurement kind -> scalar weight from weights struct.

    kindStr = string(kindStr);

    if kindStr == "probe"
        w = double(weights.w_probe);
        return;
    end

    if kindStr == "t14_mid1" || kindStr == "t14_mid2"
        w = double(weights.w14);
        return;
    end

    if kindStr == "t40_post"
        w = double(weights.w40);
        return;
    end

    % Conservative default (treat as w40)
    w = double(weights.w40);
end
