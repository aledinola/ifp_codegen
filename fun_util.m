function util = fun_util(c,gamma)
% Note: c is always positive

if gamma==1
    util = log(c);
else
    util = (c^(1-gamma)) / (1-gamma);
end

end %end function