% ---------------------------------------------
% Diffusion Propagator (fit quality / comparison)
% Adapted Version for Generating Diverse Data Sets
% Marion Menzel
% Adaptation Date: 18 April 2024
%
% Adapted to introduce variations in simulation parameters, add noise,
% and handle multiple fiber scenarios.
%
% ---------------------------------------------
    clear; close all;

% ---------------------------------------------
% 1. Define simulation parameters
% ---------------------------------------------
    simsize = 33; % simulate a larger r-space than can be sampled
bmax = 10000; % s/mm^2
bigDelta = 66E-3; % sec
smallDelta = 60E-3; % sec
gmax = 40E-3; % T/m
gamma = 2*pi*42.57E6; % Hz/T for proton
qmax = gamma/2/pi*smallDelta*gmax;
qDelta = qmax*2/(simsize-1);
rFOV = 1/qDelta;
rmax = rFOV/2;

% Simulation variations
num_simulations = 100;
D_range = [1E-11, 1E-9];
FA_range = [0.01, 0.99];
alpha_range = [0, 360]; % degrees
beta_range = [0, 360]; % degrees

% ---------------------------------------------
% 2. Generate and process data for multiple simulations
% ---------------------------------------------
    results = cell(num_simulations, 1);
for i = 1:num_simulations
% Varying diffusion coefficients and fractional anisotropy
FA = FA_range(1) + (FA_range(2) - FA_range(1)) * rand();
Dxx = D_range(1) + (D_range(2) - D_range(1)) * rand();
Dyy = Dxx * (1-FA) / sqrt(2);
Dzz = Dxx * (1-FA) / sqrt(2);

% Varying fiber fractions
f1 = rand();
f2 = 1 - f1;

% Varying rotation angles
alpha = alpha_range(1) + (alpha_range(2) - alpha_range(1)) * rand();
beta = beta_range(1) + (beta_range(2) - beta_range(1)) * rand();

% Grid generation for r-space and q-space
[X, Y, Z] = ndgrid(linspace(-rmax, rmax, simsize));
[X1, Y1, Z1] = rotateGrid(X, Y, Z, alpha, beta);

% r-space and q-space calculation
rspace = 1/sqrt(((4*pi*bigDelta)^3)*Dxx*Dyy*Dzz) * exp(-(X1.*X1/(4*Dxx*bigDelta) + Y1.*Y1/(4*Dyy*bigDelta) + Z1.*Z1/(4*Dzz*bigDelta)));
qspace = fftshift(fftn(ifftshift(rspace)));

% Adding Gaussian noise to q-space
noise_level = 0.01; % adjust noise level accordingly
qspace = qspace + noise_level * randn(size(qspace));

% Store results
results{i} = struct('rspace', rspace, 'qspace', qspace, 'Dxx', Dxx, 'Dyy', Dyy, 'Dzz', Dzz, 'FA', FA, 'f1', f1, 'f2', f2, 'alpha', alpha, 'beta', beta);
end

% ---------------------------------------------
% 3. Save results to file
% ---------------------------------------------
    save('simulation_results.mat', 'results');

% Function to perform grid rotation
function [Xr, Yr, Zr] = rotateGrid(X, Y, Z, alpha, beta)
rad_alpha = alpha * pi/180;
rad_beta = beta * pi/180;
Xr = (X*cos(rad_alpha) + Y*sin(rad_alpha))*cos(rad_beta) + Z*sin(rad_beta);
Yr = (Y*cos(rad_alpha) - X*sin(rad_alpha))*cos(rad_beta);
Zr = Z*cos(rad_beta) - (X*cos(rad_alpha) + Y*sin(rad_alpha))*sin(rad_beta);
end
