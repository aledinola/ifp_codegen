function F = ReturnFn(aprime,a,z,r,w,gamma)

c = (1+r)*a+w*z-aprime;

F = -inf;

if c>0
    if abs(gamma - 1) < 1e-12
        F = log(c);
    else
        F = (c^(1-gamma)) / (1-gamma);
    end
end

end %end function