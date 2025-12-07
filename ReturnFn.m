function F = ReturnFn(aprime,a,z,r,w,gamma)

c = (1+r)*a+w*z-aprime;

F = -inf;

if c>0
    F = fun_util(c,gamma);
end

end %end function