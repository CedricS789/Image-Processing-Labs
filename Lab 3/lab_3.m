%% ================= Exercise 11: =================
% Adding Gaussian noise and filtering with a Gaussian filter
% =================================================
clear; close all; clc;
img = imread('lenagray.tif');

% ---- Gaussian noise + Gaussian filtering ----
img_double = im2double(img);
img_noisy = imnoise(img_double, 'gaussian', 0, 0.1);
img_ano = anisodiff(img_noisy, 70, 0.1);
h_gaussian = fspecial('gaussian', 5, sqrt(2*70/10));
img_gaussian = conv2(img_noisy, h_gaussian, 'same');

figure;
subplot(3, 2, 1);
imshow(img);
title('Original Image (lenagray.tif)');

subplot(3, 2, 2);
imshow(img_double);
title('Double Image (lenagray.tif)');

subplot(3, 2, 3);
imshow(img_noisy);
title('Noisy Image (Gaussian noise)');

subplot(3, 2, 4);
imshow(img_ano);
title('Anisotropic Diffusion Image (lenagray.tif)');

subplot(3, 2, 5);
imshow(img_gaussian);
title('Gaussian Blurred Image (lenagray.tif)');



%% ================= Exercise 12: =================
% Adding Salt & Pepper noise and filtering with a median filter
% =================================================
clear; close all; clc;
img = imread('cameraman.tif');
img_double = im2double(img);
img_noisy = imnoise(img_double, 'salt & pepper', 0.1);
median_filter = dsp.MedianFilter('WindowLength', 3);
img_median_filtered = median_filter(img_noisy);

figure;
subplot(2, 2, 1);
imshow(img);
title('Original Image (cameraman.tif)');

subplot(2, 2, 2);
imshow(img_double);
title('Double Image (cameraman.tif)');

subplot(2, 2, 3);
imshow(img_noisy);
title('Noisy Image (Salt & Pepper)');

subplot(2, 2, 4);
imshow(img_median_filtered);


%% ================= Exercise 13: =================
% Unsharp Masking
% =================================================
clear; close all; clc;

img = imread('aerial.tif');
img_double = im2double(img);

% ---- Histogram Stretching ----
min_val = min(img_double(:));
max_val = max(img_double(:));
img_adjusted = imadjust(img_double, [min_val max_val], [0 1], 1);

% ---- Unsharp Masking ----
img_sharpened = imsharpen(img_adjusted);

% ---- Plots ----
figure;
subplot(3, 2, 1);
imshow(img_double);
title('Original Image Double (aerial.tif)');

subplot(3, 2, 2);
imhist(img_double, 64);
title('Histogram of Original (64 bins)');

subplot(3, 2, 3);
imshow(img_adjusted);
title('Histogram Stretched Image (aerial.tif)');

subplot(3, 2, 4);
imhist(img_adjusted, 64);
title('Histogram of Stretched (64 bins)');

subplot(3, 2, 5);
imshow(img_sharpened);
title('Sharpened Image (aerial.tif)');

subplot(3, 2, 6);
imhist(img_sharpened, 64);
title('Histogram of Sharpened (64 bins)');

figure;
subplot(1, 2, 1);
imshow(img_adjusted);
title('Histogram Stretched Image (aerial.tif)');

subplot(1, 2, 2);
imshow(img_sharpened);
title('Sharpened Image (aerial.tif)');



%% ================= Exercise 14: =================
% Unsharp Masking Manually
% =================================================
clear; close all; clc;

img = imread('aerial.tif');
img_double = im2double(img);

% ---- Histogram Stretching ----
min_val = min(img_double(:));
max_val = max(img_double(:));
img_adjusted = imadjust(img_double, [min_val max_val], [0 1], 1);

% ---- Unsharp Masking ----
sharp = 1/9 * [-1 -1 -1; -1 8 -1; -1 -1 -1];        % Laplacian kernel
img_sharpened_1 = imsharpen(img_adjusted);
g = conv2(img_adjusted, sharp, 'same');               % High-pass filtered image
img_sharpene_2 = img_adjusted + 1*g;                  % Unsharp Masking

% ---- Plots ----
figure;
subplot(3, 2, 1);
imshow(img_adjusted);
title('Histogram Stretched Image (aerial.tif)');

subplot(3, 2, 2);
imhist(img_adjusted, 64);
title('Histogram of Stretched (64 bins)');

subplot(3, 2, 3);
imshow(img_sharpened_1);
title('Sharpened Image (aerial.tif) - imsharpen()');

subplot(3, 2, 4);
imhist(img_sharpened_1, 64);
title('Histogram of Sharpened (64 bins) - imsharpen()');

subplot(3, 2, 5);
imshow(img_sharpene_2);
title('Sharpened Image (aerial.tif) - Using Laplacian kernel');

subplot(3, 2, 6);
imhist(img_sharpene_2, 64);
title('Histogram of Sharpened (64 bins) - Using Laplacian kernel');

figure;
subplot(1, 2, 1);
imshow(img_sharpened_1);
title('Sharpened Image (aerial.tif) - imsharpen()');

subplot(1, 2, 2);
imshow(img_sharpene_2);
title('Sharpened Image (aerial.tif) - Using Laplacian kernel');


%% ================= Exercise 15: =================
% Gaussian Noise + Unsharp Masking Manually 
% =================================================
clear; close all; clc;
clear; close all; clc;

img = imread('aerial.tif');
img_double = im2double(img);

% ---- Histogram Stretching ----
min_val = min(img_double(:));
max_val = max(img_double(:));
img_adjusted = imadjust(img_double, [min_val max_val], [0 1], 1);
img_noisy = imnoise(img_adjusted, 'gaussian', 0, 0.1); % Gaussian noise

% ---- Unsharp Masking of a Noisy Image ----
sharp = 1/9 * [-1 -1 -1; -1 8 -1; -1 -1 -1];        % Laplacian kernel
uniform = 1/9 * ones(3, 3);                         % Uniform kernel
g = conv2(img_noisy, sharp, 'same');               % High-pass filtered image
img_sharpened_2 = img_noisy + 1*g;               % Unsharp Masking
img_sharpened_3 = conv2(img_noisy, uniform, 'same') + 1*g; % Unsharp Masking with Gaussian noise + uniform filtering

figure;
subplot(3, 2, 1);
imshow(img_noisy);
title('Noisy Image (Gaussian noise)');

subplot(3, 2, 2);
imhist(img_noisy, 64);
title('Histogram of Noisy Image (64 bins)');

subplot(3, 2, 3);
imshow(img_sharpened_2);
title('Sharpened Noisy Image (aerial.tif) - Using Laplacian kernel Without uniform filtering');

subplot(3, 2, 4);
imhist(img_sharpened_2, 64);
title('Histogram of Sharpened Noisy Image (64 bins) - Using Laplacian kernel Without uniform filtering');

subplot(3, 2, 5);
imshow(img_sharpened_3);
title('Sharpened Noisy Image (aerial.tif) - Using Laplacian kernel With uniform filtering');

subplot(3, 2, 6);
imhist(img_sharpened_3, 64);
title('Histogram of Sharpened Noisy Image (64 bins) - Using Laplacian kernel With uniform filtering');

figure;
subplot(1, 2, 1);
imshow(img_sharpened_2);
title('Sharpened Noisy Image (aerial.tif) - Using Laplacian kernel Without uniform filtering');

subplot(1, 2, 2);
imshow(img_sharpened_3);
title('Sharpened Noisy Image (aerial.tif) - Using Laplacian kernel With uniform filtering');


%% ================= Exercise 16: =================
% Deblurring an image using Wiener filter
% =================================================
clear; close all; clc;
Ioriginal = imread("cameraman.tif");
Ioriginal = im2double(Ioriginal);

PSF = fspecial('motion', 21, 11); % Point Spread Function (PSF)
Iblurred = imfilter(Ioriginal, PSF, 'conv', 'circular'); % Blurred image

wr1 = deconvwnr(Iblurred, PSF); % Wiener deconvolution with noise ratio 0.01


% ---- Plots ----
figure;
subplot(2, 2, 1);
imshow(Ioriginal);
title('Original Image (cameraman.tif)');

subplot(2, 2, 2);
imshow(Iblurred);
title('Blurred Image (cameraman.tif)');

subplot(2, 2, 3);
imshow(wr1);
title('Wiener Filtered Image (cameraman.tif)');