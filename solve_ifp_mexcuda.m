function [V,policy_idx] = solve_ifp_mexcuda(n_a,n_z,a_grid,z_grid,pi_z,Params,vfoptions)

% ifp_vfi_mexcuda.m
% Income Fluctuation Problem solved by VFI with:
%   - loops only
%   - early exit when c <= 0
%   - EV computed via matrix multiplication for speed
%   - utility function pre-defined (no branching in iterations)
% TODO:
%   - Precompute util_fun(a',a,z) before the VFI while loop

%% Parameters

tol = vfoptions.tol;
maxiter = vfoptions.maxiter;
verbose = vfoptions.verbose;

r = Params.r;
w = Params.w;
beta = Params.beta;
gamma = Params.gamma;

%% Allocate
V  = zeros(n_a, n_z);
Vn = zeros(n_a, n_z);
policy_idx = ones(n_a, n_z);

%% Utility (handle gamma = 1 here)
if abs(gamma - 1) < 1e-12
    util_fun = @(c) log(c);
else
    util_fun = @(c) (c.^(1-gamma)) ./ (1-gamma);
end

%% ----------------------
%     Start VFI
% ----------------------
fprintf("Starting VFI with fast EV...\n");
iter = 0; diff = Inf;
tic

while iter < maxiter && diff > tol
    iter = iter + 1;

    % -----------------------------------------------------
    % FAST expectation: EV(ia,iz) = sum_{jp} V(ia,jp) * pi_z(iz,jp)
    % Equivalent: EV = V * pi_z'
    % -----------------------------------------------------
    EV = V * pi_z';     % n_a × n_z matrix

    % Loop over income states iz
    for iz = 1:n_z

        EVcol = EV(:,iz);   % n_a × 1 vector, continuation value for each a'

        % Loop over asset states a
        for ia = 1:n_a
            cash = (1+r)*a_grid(ia) + w*z_grid(iz);

            best_val = -Inf;
            best_iap = 1;

            % Loop over next-period assets a'
            for iap = 1:n_a
                aprime = a_grid(iap);
                c = cash - aprime;
                if c > 0
                    val = util_fun(c) + beta * EVcol(iap);

                    if val > best_val
                        best_val = val;
                        best_iap = iap;
                    end
                else
                    % Early break: remaining aprime all too large
                    break
                end
            end %end a'

            Vn(ia,iz) = best_val;
            policy_idx(ia,iz) = best_iap;
        end %end a
    end %end z

    diff = max(abs(Vn(:) - V(:)));
    V = Vn;

    if verbose==1
        if mod(iter,20)==0 || iter == 1
            fprintf("Iter %4d: diff = %.8e\n", iter, diff);
        end
    end
end %end while

t_elapsed = toc;
fprintf("Done: %d iterations, diff = %.3e, time = %.2f sec\n", iter, diff, t_elapsed);

end %end function