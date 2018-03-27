%% COMPGV15: Project

%% Read in K-Space Data set from http://ymk.k-space.org/medical_image_reconstruction.htm
% inverse fast fourier trasnform
MIR_2010_partialFourierdata = importdata('MIR_2010_partialFourierdata.txt');
%reshape data into 256x256 image
ksp = reshape(MIR_2010_partialFourierdata(:,1) + 1i*MIR_2010_partialFourierdata(:,2), 256, 256);
%display k-space
figure(1);
subplot(1,2,1); image(abs(ksp)*1e6); colormap(gray);
% inverse fast fourier transform
im = fftshift(ifft2(fftshift(ksp)));
subplot(1,2,2); imagesc(abs(im)); colormap(gray);

%% First do non-regularized least squares
% Solve minimize norm(y - Af) 