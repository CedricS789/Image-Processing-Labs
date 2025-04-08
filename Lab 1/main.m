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
image_matrix = zeros(128, 128);         % Create a 128x128 matrix of zeros (black image)
image_matrix(64-3:64+3, 34:96) = 1;     % Create a horizontal white line in the middle of the image 
imshow(image_matrix);                   % Display the image

% ---- compute the 2D DFT ----
dft_coeffs = fft2(image_matrix); disp('Size of DFT coefficients matrix:'); disp(size(dft_coeffs));

% ---- shif the zero-frequency component to the center of the secptrum ----
dft_shifted = fftshift(dft_coeffs); disp('DFT coefficients shifted!');
magnitude_spectrum = abs(dft_shifted);

% Display using imagesc (as per Ex 4 instructions) - without log scaling
figure; % New figure for imagesc view
imagesc(magnitude_spectrum);
axis image;
colormap gray;
colorbar;
title('Exercise 4: Magnitude Spectrum (imagesc)');

% Display using surfc (as per Ex 4 instructions) - also uses magnitude
figure; % New figure for surfc view
surfc(magnitude_spectrum);
shading interp; % Makes the surface look smoother
title('Exercise 4: Magnitude Spectrum (surfc)');
xlabel('Frequency component l');
ylabel('Frequency component k');
zlabel('Magnitude');