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
   subplot(2,4,i);image(abs(sensitivity(:,:,i)*1e2)); colormap(gray);axis off
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
%clear k m n w sens sensitivity; % cleaning data not used for reconstruction
%% Conjugate Gradient Method no Regularisation
[x, t, d, ~] = conjugateGradient(@(x) A(x), a, 1e-9, 30, P, a);
err = x-ref;
fprintf('\t-> Reconstruction performed with %d iterations\n',t);
figure;imagesc(abs(x));colormap gray;axis off;colorbar;title('reconstructed image no regularisation');
figure;imagesc(abs(err));colormap(1-gray);axis off;colorbar;title('error map in inverted gray levels');
figure;semilogy(0:1:t,d,'*-');xlabel('iteration');ylabel('residual');

%% Preconditoned Conjugate Gradient Method 1st order Tikhonov Regularisation Zeros Reference Image
% Their solution for lambda
lambda = 1e-3*max(abs(a(:)));
[x, t, d, ~] = conjugateGradient(@(x) A(x) + lambda*(x-zeros(size(x))), a, 1e-6, 30, P, a);
err = x-ref;
fprintf('\t-> Reconstruction performed with %d iterations\n',t);
figure;imagesc(abs(x));colormap gray;axis off;colorbar;title('reconstructed image (CG)');
figure;imagesc(abs(err));colormap(1-gray);axis off;colorbar;title('error map (CG) in inverted gray levels');
%figure;semilogy(0:1:t,d,'*-');xlabel('iteration');ylabel('residual');

%% Preconditioned Conjugate Gradient Method 1st order Tikhonov Regularisation No Reference Image
% Their solution for lambda
lambda = 1e-3*max(abs(a(:)));
[x, t, d, ~] = conjugateGradient(@(x) A(x) + lambda*x, a, 1e-9, 30, P, a);
err = x-ref;
fprintf('\t-> Reconstruction performed with %d iterations\n',t);
figure;imagesc(abs(x));colormap gray;axis off;colorbar;title('reconstructed image (PCG)');
figure;imagesc(abs(err));colormap(1-gray);axis off;colorbar;title('error map (PCG) in inverted gray levels');
%figure;semilogy(0:1:t,d,'*-');xlabel('iteration');ylabel('residual');

%% Total Variation Method
lambda = 1e-3*max(abs(a(:)));
[x, t, d, ~] = totalVariation(@(x) A(x), a, lambda, 1e-9, 10, P, a);
err = x-ref;
fprintf('\t-> Reconstruction performed with %d iterations\n',t);
figure;imagesc(abs(x));colormap gray;axis off;colorbar;title('reconstructed image (PCG)');
figure;imagesc(abs(err));colormap(1-gray);axis off;colorbar;title('error map (PCG) in inverted gray levels');

