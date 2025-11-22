%% Purpose: Solve income fluctuation problem with
%  (1) intrinsic GPU commands (as in VFI toolkit), and
%  (2) MEX-CUDA.
%  Compare speed and memory usage.

clear; clc; close all;

%% Set parameters

% Size of the grids
n_a = 1000;
n_z = 11;

% Structural parameters
Params.beta  = 0.96;
Params.alpha = 0.36;
Params.delta = 0.08;
Params.crra  = 3;
Params.sigma = 0.2;
Params.rho   = 0.6;

% Prices
Params.r = 1 / Params.beta - 1;

% Grid for assets
k_ss   = ((Params.r + Params.delta) / Params.alpha)^(1 / (Params.alpha - 1));
a_min  = 0;
a_max  = 10 * k_ss;
a_grid = a_min + (a_max - a_min) * (linspace(0, 1, n_a).^3)';

%% Solve model using intrinsic gpuArray (as in VFI toolkit)

t_start_gpu = tic;
[V, Policy] = solve_ifp_gpuarray(1,1);
time_gpu    = toc(t_start_gpu);

%% Solve model using MEX-CUDA

t_start_mex = tic;
[V2, Policy2] = solve_ifp_mexcuda(1,1);
time_mex      = toc(t_start_mex);

%% Compute approximation errors

errV = max(abs(V - V2), [], "all");
errP = max(abs(Policy - Policy2), [], "all");


%% Display results: timings and errors

fprintf('\n=========================================\n');
fprintf('   Income Fluctuation Problem Summary\n');
fprintf('=========================================\n');

fprintf('Grid sizes: n_a = %d, n_z = %d\n', n_a, n_z);
fprintf('\nTiming:\n');
fprintf('  gpuArray implementation : %8.4f seconds\n', time_gpu);
fprintf('  MEX-CUDA implementation : %8.4f seconds\n', time_mex);
fprintf('  Speedup (MEX / gpuArray): %8.2f x\n', time_gpu / time_mex);

fprintf('\nAccuracy (max abs. errors):\n');
fprintf('  max |V_gpu - V_mex|      : %8.3e\n', errV);
fprintf('  max |P_gpu - P_mex|      : %8.3e\n', errP);
fprintf('=========================================\n\n');
