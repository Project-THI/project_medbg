% ---------------------------------------------
% Diffusion Propagator (fit quality / comparison)
% Marion Menzel
% 05 July 2010 (include rotation)
% 05 Aug 2010
% 01 Apr 2024 (subset for KI Project)   
%
% see e.g. Lin et al., ChemPhysLett 335 (2001) 249-256. 
%
% TODO: 
% - scaling of X,Y,Z axis after ifft -> qspace range missing
% - alternative sampling schemes (non-cartesian)
% - add noise
% ---------------------------------------------
% clear; close all

% ---------------------------------------------
%1. Define simulation parameters
% ---------------------------------------------
simsize       = 33; %simulate a larger r-space than can be sampled
ssize         = 33; %simulate a larger r-space than can be sampled
bmax = 10000; %s/mm^2
bigDelta = 66E-3; %sec. See van Wedeen, MRM 2005, or mixing time
smallDelta=60E-3; %sec. See van Wedeen, MRM 2005
gmax = 40E-3; %T/m
gamma = 2*pi*42.57E6; %Hz/T for proton
qmax = gamma/2/pi*smallDelta*gmax; %By definition
qDelta=qmax*2/(simsize-1); %By definition
rFOV=1/qDelta; %By definition
rmax=rFOV/2;

% VARIATIONAL PARAMETER FA - can vary between [0.01 - 0.99] in case you
% would like to fix the propagator shape to a "cigar type", meaning one
% long axis and two short axis
FA=0.85; %Assumed FA


%Diffusion Coefficients for 1st Fiber
%VARIATIONAL PARAMETERS Dxx / Dyy / Dzz (diffusion coefficients) can vary between [1E-11 and 1E-8]
Dxx1        = 2E-9; % [m^2 * s^-1]  Water @ 20°C. Diagonal components are assumed to be very small. See Konstantinos AJNR 02
Dyy1        = Dxx1*(1-FA)/sqrt(2); % [m^2 * s^-1] Similar to Tuch MRM 2004
Dzz1        = Dxx1*(1-FA)/sqrt(2); % [m^2 * s^-1] Similar to Tuch MRM 2004

%Diffusion Coefficients for 2nd Fiber
Dxx2        = 2E-9; % [m^2 * s^-1]  
Dyy2        = Dxx2*(1-FA)/sqrt(2); % [m^2 * s^-1] Similar to Tuch MRM 2004
Dzz2        = Dxx2*(1-FA)/sqrt(2); % [m^2 * s^-1] Similar to Tuch MRM 2004

%Diffusion Coefficients for 3rd Fiber
Dxx3        = 2E-9; % [m^2 * s^-1] 
Dyy3        = Dxx2*(1-FA)/sqrt(2); % [m^2 * s^-1] Similar to Tuch MRM 2004
Dzz3        = Dxx2*(1-FA)/sqrt(2); % [m^2 * s^-1] Similar to Tuch MRM 2004

%Relative fiber fraction in combined fibers 
% VARIATIONAL PARAMETER, can vary between [0.01 and 0.99]
f1=0.5; 
f2=1-f1;

% in case of more than two fibers per voxel please define:
% f1 = 0.1
% f2 = 0.2
% f3 = 1- (f1+f2

% ---------------------------------------------
%2. Generate data grid in 3D for sampled data (both q-space and r-space)
% ---------------------------------------------
[X,Y,Z] = ndgrid(linspace(-rmax,rmax,simsize),linspace(-rmax,rmax,simsize),linspace(-rmax,rmax,simsize));
[qX,qY,qZ] = ndgrid(linspace(-qmax,qmax,simsize),linspace(-qmax,qmax,simsize),linspace(-qmax,qmax,simsize));

% define rotation angles (alpha and beta) for in and out of plane rotation
% - not a complete set of Euler Angles though
alp1 = 10;
bet1=0;
X1=(X*cos(alp1/180*pi)+Y*sin(alp1/180*pi))*cos(bet1/180*pi)+Z*sin(bet1/180*pi);
Y1=(Y*cos(alp1/180*pi)-X*sin(alp1/180*pi))*cos(bet1/180*pi);
Z1=Z*cos(bet1/180*pi)-(X*cos(alp1/180*pi)+Y*sin(alp1/180*pi))*sin(bet1/180*pi);

alp2 = 80;
bet2=0;
X2=(X*cos(alp2/180*pi)+Y*sin(alp2/180*pi))*cos(bet2/180*pi)+Z*sin(bet2/180*pi);
Y2=(Y*cos(alp2/180*pi)-X*sin(alp2/180*pi))*cos(bet2/180*pi);
Z2=Z*cos(bet2/180*pi)-(X*cos(alp2/180*pi)+Y*sin(alp2/180*pi))*sin(bet2/180*pi);

% ---------------------------------------------
%3. Combine two fibres with Gaussian mixture model
% ---------------------------------------------
% simulate r-space data (propagator of diffusion)
rspace1 = 1/sqrt(((4*pi*bigDelta)^3)*Dxx1*Dyy1*Dzz1)*exp(-(X1.*X1/(4*Dxx1*bigDelta)+Y1.*Y1/(4*Dyy1*bigDelta)+Z1.*Z1/(4*Dzz1*bigDelta)));
rspace2 = 1/sqrt(((4*pi*bigDelta)^3)*Dxx2*Dyy2*Dzz2)*exp(-(X2.*X2/(4*Dxx2*bigDelta)+Y2.*Y2/(4*Dyy2*bigDelta)+Z2.*Z2/(4*Dzz2*bigDelta)));
rspace3 = 1/sqrt(((4*pi*bigDelta)^3)*Dxx3*Dyy3*Dzz3)*exp(-(X2.*X2/(4*Dxx3*bigDelta)+Y2.*Y2/(4*Dyy2*bigDelta)+Z2.*Z2/(4*Dzz2*bigDelta)));

%example combination of two fibers with respective fractions
%-> rspace: this is the output result (perfect ground truth)
rspace = f1*rspace1 + f2*rspace2; 
save('rspace.mat','rspace');

 %calculate q-space by fast fourier transform -> this is the input data (this is perfect input without any noise/artefacts)
qspace = fftshift(fftn(ifftshift(rspace)));
save('qspace.mat','qspace');

% qspace1 = fftshift(fftn(ifftshift(rspace1)));
% qspace2 = fftshift(fftn(ifftshift(rspace2)));
% qspace3 = fftshift(fftn(ifftshift(rspace3)));

