%% COMPGV15: Project
% Using BIG http://bigwww.epfl.ch/algorithms/mri-reconstruction/
addpath(genpath('BIG'));

% Simulation type
%simu = 'analytical';
simu = 'rasterized';
%% Define MRI Setup
mxsize = 256*[1,1];
R = 4;
f_sampling = 1/1.3;
traj = 'cartesian';
%traj = 'spiral';
snr_hf = .7; % snr for the data that are outside the disc with radius 90% the highest frequency sampled
Ncoils = 8;
FOV = 0.28; % FOV width
pixelsize = FOV./mxsize;

%% Define phantom
disp('Defining analytical phantom');
DefineBrain;
Brain.FOV = FOV*[1, 1];
ref = RasterizePhantom(Brain,mxsize);
support = (ref>5e-3*max(ref(:)));
support = imdilate(support,strel('disk',5));

%% Coils simulation
disp('Simulating coils');
sensitivity = GenerateSensitivityMap( FOV, pixelsize, Ncoils, .07, .17);
% plot coil sensitivities
for i=1:size(sensitivity,3)
   subplot(2,4,i);image(abs(sensitivity(:,:,i)*1e2)); colormap(gray);
end

%% K-SPACE
disp('Simulating Cartesian K-space traj');
w = GenerateCartesianTraj(FOV, pixelsize, f_sampling, R);

%% MR SIMULATIONS
m = zeros(size(w,1),Ncoils);
switch simu
    case 'analytical'
        disp('Simulating analytical MR data');
        sens=cell(1,Ncoils);
        for indCoil = 1:Ncoils
            sens{indCoil} = SensFitting(sensitivity(:,:,indCoil),'sinusoidal',6,support);
            m(:,indCoil) = MRDataAnalytical(Brain, sens{indCoil}, w);
            [~,sensitivity(:,:,indCoil)] = RasterizePhantom(Brain,mxsize,sens{indCoil});
        end
        m = prod(mxsize)*m;
    case 'rasterized'
        disp('Simulating rasterized MR data');
        refinement = 3;
        im_rast = RasterizePhantom(Brain,refinement*mxsize);
        sens = GenerateSensitivityMap( FOV, FOV./mxsize/refinement, Ncoils, .07, .17);
        for indCoil = 1:Ncoils
            m(:,indCoil) = MRDataRasterized(sens(:,:,indCoil).*im_rast, w, FOV)/refinement^ndims(im_rast);
        end
end

%% Add noise
disp('Simulating noise');
n = SimulateNoise(m,w,snr_hf,true);
m = m + n;

%% Prepare Data for Recon
disp('Preparing data for reconstruction');
k = TrajInGridUnits(w, FOV, mxsize);
figure;plot(k(:,1),k(:,2),'.');axis square;title('k-space traj in grid units.')
[a,A,P] = Prepare4Recon(m, k, sensitivity, support);
x = a./P./P; % starting point
% clear k m n w sens sensitivity; % cleaning data not used for reconstruction

