function [V,Policy] = solve_ifp_gpuarray(n_a,n_z,a_grid,z_grid,pi_z,Params,vfoptions)

% ifp_vfi_gpuarray.m
% Income Fluctuation Problem solved by VFI as the VFI toolkit would do it :)
%   - vectorized code only
%   - use gpuArray and arrayfun 
%   - EV computed via matrix multiplication for speed
%   - No Howard improvement here (neither in MEX CUDA)
% TODO:
%   - Ask Robert if this is a fair implementation of what toolkit does
%   behind the scenes. "Fair" means that I want to give the toolkit the 
%   best chance

tol = vfoptions.tol;
maxiter = vfoptions.maxiter;
verbose = vfoptions.verbose;

% Move arrays to GPU and reshape as (a',a,z)
aprime_vals = gpuArray(a_grid);
a_vals      = reshape(aprime_vals,[1,n_a,1]);
z_vals      = reshape(gpuArray(z_grid),[1,1,n_z]);

ParamCell = cell(3,1);
ParamCell{1} = Params.r;
ParamCell{2} = Params.w;
ParamCell{3} = Params.gamma;
beta         = Params.beta;

F_matrix = arrayfun(@ReturnFn,aprime_vals,a_vals,z_vals,ParamCell{:});

V0 = zeros(n_a,n_z,"gpuArray");

err = tol+1.0;
iter = int32(1);

while err>tol && iter<=maxiter

    % EV(a',z) = sum_{z'} V0(a',z')*P(z,z')
    EV = V0*transpose(pi_z);

    EV_long   = reshape(EV,[n_a,1,n_z]);
    % entireRHS(a',a,z)
    entireRHS = F_matrix + beta*EV_long;

    [max_val,max_ind] = max(entireRHS,[],1);

    V      = reshape(max_val,[n_a,n_z]);
    Policy = reshape(max_ind,[n_a,n_z]);

    err = max(abs(V-V0),[],"all");

    if verbose==1
        if mod(iter,20)==0 || iter == 1
            fprintf("Iter %4d: err = %.8e\n", iter, err);
        end
    end

    % Update
    V0 = V;
    iter = iter+1;

end %end while

if err>tol
    warning('VFI did not converge!')
end

end %end function solve_ifp_gpuarray