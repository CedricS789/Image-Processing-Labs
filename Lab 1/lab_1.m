clear; close all; clc;

%% ================= Exercise 1 - Indexed Image Conversion & YUV Check ==================
%
% Goal: To practice reading an indexed image, converting it to RGB and greyscale, 
% and verifying the greyscale conversion process using the YUV formula
%
% ==========================================================================

[aind, amap] = imread('trees.tif', 'TIF');  % aind contain the idices of the image, and amap contains the RGB colors
figure(1);
imshow(aind, amap);                         % Display the indexed image

rgb_image = ind2rgb(aind, amap);            % Convert indexed image to RGB
figure(2);
imshow(rgb_image);                          % Display the RGB image

idx = aind(3, 4);                           % Get the index of the pixel at (3, 4)
color_from_map = amap(idx+1, :);            % Get the color from the colormap

gray_image = rgb2gray(rgb_image);           % Convert RGB image to greyscale
figure(3);
imshow(gray_image);                         % Display the greyscale image

% ----- Check the YUV conversion -----
% The YUV conversion formula is: Y = 0.299*R + 0.587*G + 0.114*B
% YUV means : Y = Luminance, U = Blue - Luminance, V = Red - Luminance ????
Y1 = 0.299 * color_from_map(1) + 0.587 * color_from_map(2) + 0.114 * color_from_map(3); % Y from RGB
Y2 = gray_image(3, 4);                                              % Y from greyscale image
fprintf('Y from RGB: %f, Y from greyscale image: %f\n', Y1, Y2);    % Display the Y values



%% ================= Exercise 2: Exploring Colormaps ==================
%
% Goal: To understand how changing the colormap dramatically affects the visual appearance
% of an indexed image, even when the underlying index data remains the same.
%
% ==========================================================================

X = [1 2 3; 3 1 2; 2 3 1];                                          % Create a 3x3 matrix of indices
X_map_initial = [1 0 0; 0 1 0; 0 0 1];                              % Red; Green; Blue
figure; imshow(X, X_map_initial, 'InitialMagnification', 'fit');    % Display the image with the initial colormap
X_map_cmy = [0 1 1; 1 0 1; 1 1 0];                                  % Cyan; Magenta; Yellow (Order corresponds to indices 1, 2, 3 in X)
figure; imshow(X, X_map_cmy, 'InitialMagnification', 'fit');        % Display the image with the CMY colormap

% ----- Change to Greyscale -----
X_map_grey = [0 0 0; 0.5 0.5 0.5; 1 1 1];                           % Black; Grey; White (Order corresponds to indices 1, 2, 3 in X)
figure; imshow(X, X_map_grey, 'InitialMagnification', 'fit');       % Display the image with the greyscale colormap


%% ================= Exercise 3: 1D Filtering (Convolution) ==================
%
% Goal: To implement a basic 1D filtering operation (specifically, convolution with an averagind filter)
% from scratch in MATLAB, understanding how filters modify signals and how to handle boundaries
%
% ============================================================================
clear; close all; clc;

s   = rand(1, 3)            % Create a random signal of length 20
h   = 1/3 * [1 1 1]         % Average filter kernel
r1  = my_convo(s, h)        % Apply the filter using the custom function
r2  = conv(s, h, 'same')    % Apply the filter using MATLAB's built-in function
r1  == r2                   % Check if the results are the same

function r = my_convo(s, h)
    s = s(:)';                    % Ensure s is a row vector
    h = h(:)';                    % Ensure h is a row vector
    
    % The exercise statement ask that the filter should have an odd number of sample
    % Therefore we check first the length of our samples 
    N = length(s);                % Length of the signal
    M = length(h);                % Length of the filter
    r = zeros(1, N);              % Initialize the output signal with zeros

    % Check if filter length is odd (required for simple center alignment)
    % The mod(M, 2) function returns the remainder after dividing M by 2.
    % If the remainder is 0, the length is even.
    if mod(M, 2) == 0
        error('Filter length must be odd for this simple implementation.');
    end

    % Calculate the padding size based on the filter length M so the length of the output signal is the same as the input signal

    pad_size = (M - 1) / 2                                                 % Ensure the centering when filtering
    s_padded = [zeros(1, pad_size), s, zeros(1, pad_size)]                 % Pad the signal with zeros on both sides

    % using pad_size ensures that in the loop, we don't overflow, since the convolution will go from
    % 1 to N, the middle sample of the filter will be aligned with the loop index i
    % We need to loop over each element of the original signal to compute the output
    for i = 1:N
        s_segment = s_padded(i:i+M-1);                                     % Extract the segment of the signal to convolve with the filter
        r(i) = dot(s_segment, h);         
    end
end


%% ================= Exercise 4: Discrete Fourier Transform (DFT) - Spectrum Analysis =============
%
% To compute the 2D DFT of a simple image, visualize its spectrum, understand the relationship between spatial features and frequency components, 
% and observe the rotation property of the DFT
%
% ============================================================================
clear; close all; clc;

% ---- Create the original image and its rotations ----
image_matrix = zeros(128, 128);                     % Create a 128x128 matrix of zeros (black image)
image_matrix(64-3:64+3, 34:96) = 1;                 % Create a horizontal white line in the middle of the image
image_matrix_rot_45 = imrotate(image_matrix, 45);   % Rotate the image by 45 degrees
image_matrix_rot_60 = imrotate(image_matrix, 60);   % Rotate the image by 60 degrees
image_matrix_rot_90 = imrotate(image_matrix, 90);   % Rotate the image by 90 degrees

% ---- Display all original and rotated images in one figure ----
figure;                                             % New figure for the images
subplot(2, 2, 1);
imshow(image_matrix);
title('Original (0 degrees)');

subplot(2, 2, 2);
imshow(image_matrix_rot_45);
title('Rotated 45 degrees');

subplot(2, 2, 3);
imshow(image_matrix_rot_60);
title('Rotated 60 degrees');

subplot(2, 2, 4);
imshow(image_matrix_rot_90);
title('Rotated 90 degrees');
sgtitle('Original and Rotated Images');             % Overall title for the figure (requires R2018b or later)


% ---- compute the 2D DFT for all images ----
dft_coeffs = fft2(image_matrix);
dft_coeffs_rot_45 = fft2(image_matrix_rot_45);
dft_coeffs_rot_60 = fft2(image_matrix_rot_60);
dft_coeffs_rot_90 = fft2(image_matrix_rot_90);


% ---- shift the zero-frequency component to the center of the spectrum for all ----
dft_shifted = fftshift(dft_coeffs);
dft_shifted_rot_45 = fftshift(dft_coeffs_rot_45);
dft_shifted_rot_60 = fftshift(dft_coeffs_rot_60);
dft_shifted_rot_90 = fftshift(dft_coeffs_rot_90);

% ---- Calculate magnitude spectrum for all ----
magnitude_spectrum = abs(dft_shifted);
magnitude_spectrum_rot_45 = abs(dft_shifted_rot_45);
magnitude_spectrum_rot_60 = abs(dft_shifted_rot_60);
magnitude_spectrum_rot_90 = abs(dft_shifted_rot_90);

% ---- Display all magnitude spectra using imagesc in one figure ----
figure;                                             % New figure for imagesc view
subplot(2, 2, 1);
imagesc(magnitude_spectrum);
axis image; colormap gray; colorbar;
title('Magnitude Spectrum (0 deg)');

subplot(2, 2, 2);
imagesc(magnitude_spectrum_rot_45);
axis image; colormap gray; colorbar;
title('Magnitude Spectrum (45 deg)');

subplot(2, 2, 3);
imagesc(magnitude_spectrum_rot_60);
axis image; colormap gray; colorbar;
title('Magnitude Spectrum (60 deg)');

subplot(2, 2, 4);
imagesc(magnitude_spectrum_rot_90);
axis image; colormap gray; colorbar;
title('Magnitude Spectrum (90 deg)');
sgtitle('Magnitude Spectra (imagesc view)');            % Overall title

% ---- Display all magnitude spectra using surfc in one figure ----
figure; % New figure for surfc view
subplot(2, 2, 1);
surfc(magnitude_spectrum);
title('Magnitude Spectrum (0 deg)');
xlabel('Frequency component l'); ylabel('Frequency component k'); zlabel('Magnitude');

subplot(2, 2, 2);
surfc(magnitude_spectrum_rot_45);
title('Magnitude Spectrum (45 deg)');
xlabel('Frequency component l'); ylabel('Frequency component k'); zlabel('Magnitude');

subplot(2, 2, 3);
surfc(magnitude_spectrum_rot_60);
title('Magnitude Spectrum (60 deg)');
xlabel('Frequency component l'); ylabel('Frequency component k'); zlabel('Magnitude');

subplot(2, 2, 4);
surfc(magnitude_spectrum_rot_90);
title('Magnitude Spectrum (90 deg)');
xlabel('Frequency component l'); ylabel('Frequency component k'); zlabel('Magnitude');
sgtitle('Magnitude Spectra (surfc view)');          % Overall title


%% ================= Exercise 5: =================
%
% ================================================
clear; close all; clc;

% Read the grayscale images
A = imread('mandrill.tif', 'TIF');
B = imread('zebra.tif', 'TIF');

% Compute the DFT
dft_A = fft2(A);
dft_B = fft2(B);

% Extract modulus (magnitude) and phase values (use abs and angle)
mag_A   = abs(dft_A);
phase_A = angle(dft_A);
mag_B   = abs(dft_B);
phase_B = angle(dft_B);

% Calculate shifted magnitude spectra (use fftshift)
mag_A_shifted = fftshift(mag_A);
mag_B_shifted = fftshift(mag_B);

% Represent the log10 of the modulus & shifted variants (use log10 of 1+modulus, and imshow(..., []))
% Using subplot for better organization
figure('Name', 'Log Magnitude Spectra');

% Image A: Unshifted vs Shifted
subplot(2, 2, 1);
imshow(log10(1 + mag_A), []); % Unshifted
title('Log Mag Spectrum A (Unshifted)');
colorbar;

subplot(2, 2, 2);
imshow(log10(1 + mag_A_shifted), []); % Shifted
title('Log Mag Spectrum A (Shifted)');
colorbar;

% Image B: Unshifted vs Shifted
subplot(2, 2, 3);
imshow(log10(1 + mag_B), []); % Unshifted
title('Log Mag Spectrum B (Unshifted)');
colorbar;

subplot(2, 2, 4);
imshow(log10(1 + mag_B_shifted), []); % Shifted
title('Log Mag Spectrum B (Shifted)');
colorbar;

sgtitle('Unshifted vs Shifted Log Magnitude Spectra'); % Overall title (R2018b+)

% Create two new DFTs by swapping magnitude and phase (use 1i)
new_dft1 = mag_A .* exp(1i * phase_B); % A's Magnitude + B's Phase
new_dft2 = mag_B .* exp(1i * phase_A); % B's Magnitude + A's Phase

% Compute the inverse transform (use ifft2)
recon_image1 = round(ifft2(new_dft1));
recon_image2 = round(ifft2(new_dft2));

% Show the resulting pictures (use round on ifft coefficients, and imshow(..., ()))
% Using subplot for better organization
figure('Name', 'Reconstructed Images');

subplot(1, 2, 1);
imshow(recon_image1, []);
title('Recon 1 (Mag A + Phase B)');

subplot(1, 2, 2);
imshow(recon_image2, []); 
title('Recon 2 (Mag B + Phase A)');

sgtitle('Reconstructed Images from Swapped Components'); % Overall title (R2018b+)

% The final step is to observe these reconstructions and comment on the results.