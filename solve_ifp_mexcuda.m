function [V,policy_idx] = solve_ifp_mexcuda(n_a,n_z,a_grid,z_grid,pi_z,Params,vfoptions)
% solve_ifp_mexcuda.m
% Income Fluctuation Problem solved by VFI with:
%   - loops only
%   - early exit when c <= 0
%   - EV computed via matrix multiplication for speed
%   - utility function pre-defined (no branching in iterations)
%   - cash precomputed as a 2D matrix
%
% REFERENCES:
%   - Buera & Shin GPU codes

%% Parameters
tol     = vfoptions.tol;
maxiter = vfoptions.maxiter;
verbose = vfoptions.verbose;

r     = Params.r;
w     = Params.w;
beta  = Params.beta;
gamma = Params.gamma;

%% Allocate memory
V          = zeros(n_a, n_z);
Vn         = zeros(n_a, n_z);
policy_idx = ones(n_a, n_z);

%% Precompute cash(ia, iz) = (1+r)a + wz
cash = zeros(n_a, n_z);
for iz = 1:n_z
    for ia = 1:n_a
        cash(ia, iz) = (1+r)*a_grid(ia) + w*z_grid(iz);
    end
end

%% ----------------------
%       Start VFI
% ----------------------

iter = 0; 
diff = tol+1.0;

while iter < maxiter && diff > tol
    iter = iter + 1;

    % Fast expectation: EV = V * pi_z'
    EV = V * pi_z';

    % Loop over income states
    for iz = 1:n_z

        EVcol = EV(:,iz);

        % Loop over asset states
        for ia = 1:n_a

            cash_az = cash(ia, iz);

            best_val = -Inf;
            best_iap = 1;

            % Loop over next-period assets
            for iap = 1:n_a

                aprime = a_grid(iap);
                c      = cash_az - aprime;

                if c > 0
                    val = fun_util(c,gamma) + beta * EVcol(iap);

                    if val > best_val
                        best_val = val;
                        best_iap = iap;
                    end
                else
                    break
                end
            end

            Vn(ia,iz)         = best_val;
            policy_idx(ia,iz) = best_iap;
        end
    end

    % --- UPDATED LINE ---
    diff = max(abs(Vn - V), [], 'all');

    V = Vn;

    if verbose == 1
        if mod(iter,20) == 0 || iter == 1
            fprintf("Iter %4d: diff = %f \n", iter, diff);
        end
    end
end %end while

end %end function
