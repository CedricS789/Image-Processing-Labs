%% ================= Exercise 6: =================
% Displaying images and their histograms using subplots with titles
% ================================================
clear; close all; clc;

% --- Process lenad.tif and lenal.tif ---
% Load images
img_lenad = imread('lenad.tif');
img_lenal = imread('lenal.tif');

% Convert to double
img_lenad_double = im2double(img_lenad);
img_lenal_double = im2double(img_lenal);

% Show images and histograms in subplots
figure; % Create figure for first pair

subplot(2, 2, 1); % Position for lenad image
imshow(img_lenad_double);
title('lenad.tif (double)'); % Title for this subplot

subplot(2, 2, 2); % Position for lenad histogram
imhist(img_lenad_double, 64);
title('Histogram of lenad.tif (64 bins)'); % Title for this subplot

subplot(2, 2, 3); % Position for lenal image
imshow(img_lenal_double);
title('lenal.tif (double)'); % Title for this subplot

subplot(2, 2, 4); % Position for lenal histogram
imhist(img_lenal_double, 64);
title('Histogram of lenal.tif (64 bins)'); % Title for this subplot


% --- Process lenagray.tif and lenalc.tif ---
% Load images
img_lenagray = imread('lenagray.tif');
img_lenalc = imread('lenalc.tif');

% Convert to double
img_lenagray_double = im2double(img_lenagray);
img_lenalc_double = im2double(img_lenalc);

% Show images and histograms in subplots
figure; % Create figure for second pair

subplot(2, 2, 1); % Position for lenagray image
imshow(img_lenagray_double);
title('lenagray.tif (double)'); % Title for this subplot

subplot(2, 2, 2); % Position for lenagray histogram
imhist(img_lenagray_double, 64);
title('Histogram of lenagray.tif (64 bins)'); % Title for this subplot

subplot(2, 2, 3); % Position for lenalc image
imshow(img_lenalc_double);
title('lenalc.tif (double)'); % Title for this subplot

subplot(2, 2, 4); % Position for lenalc histogram
imhist(img_lenalc_double, 64);
title('Histogram of lenalc.tif (64 bins)'); % Title for this subplot



%% ================= Exercise 7: Histogram Stretching =================
% Applying linear histogram stretching to baboon.tif
% ================================================================
clear; close all; clc;

% --- Load and Prepare Original Image ---
% Load the image
img_orig = imread('baboon.tif');

% Convert to double format
img_double = im2double(img_orig);

% --- Perform Histogram Stretching ---
% Find min and max intensity values of the original double image
min_val = min(img_double(:));
max_val = max(img_double(:));

% Apply linear histogram stretching using imadjust with gamma = 1
% Maps values from [min_val, max_val] in the input
% to [0, 1] in the output.
img_stretched = imadjust(img_double, [min_val max_val], [0 1], 1);      % This is the Key Command

% --- Display Results ---
figure;

% Display original image
subplot(2, 2, 1);
imshow(img_double);
title('Original baboon.tif (double)');

% Display histogram of original image (64 bins)
subplot(2, 2, 2);
imhist(img_double, 64);
title('Histogram of Original (64 bins)');
axis tight; % Adjust axis

% Display stretched image
subplot(2, 2, 3);
imshow(img_stretched);
title('Stretched baboon.tif');

% Display histogram of stretched image (64 bins)
subplot(2, 2, 4);
imhist(img_stretched, 64);
title('Histogram of Stretched (64 bins)');
axis tight; % Adjust axis

sgtitle('Exercise 7: Histogram Stretching'); % Add overall title



%% ================= Exercise 8: Gamma Correction =================
% Applying gamma correction to lenagray.tif
% ============================================================
clear; close all; clc;

% --- Load and Prepare Original Image ---

% Load the image
img_orig = imread('lenagray.tif');

% Convert to double format
img_double = im2double(img_orig);

% --- Find Min/Max for Normalization ---
min_val = min(img_double(:));
max_val = max(img_double(:));

% --- Apply Gamma Correction ---

% Gamma = 3
gamma1 = 3;
img_gamma1 = imadjust(img_double, [min_val max_val], [0 1], gamma1);

% Gamma = 0.33
gamma2 = 0.33;
img_gamma2 = imadjust(img_double, [min_val max_val], [0 1], gamma2);

% --- Display Results ---

figure; % Create a new figure window

% Original Image and Histogram
subplot(3, 2, 1);
imshow(img_double);
title('Original lenagray.tif (double)');

subplot(3, 2, 2);
imhist(img_double, 64);
title('Histogram of Original (64 bins)');

% Gamma = 3 Image and Histogram
subplot(3, 2, 3);
imshow(img_gamma1);
title(['Gamma = ', num2str(gamma1)]);

subplot(3, 2, 4);
imhist(img_gamma1, 64);
title(['Histogram, Gamma = ', num2str(gamma1)]);

% Gamma = 0.33 Image and Histogram
subplot(3, 2, 5);
imshow(img_gamma2);
title(['Gamma = ', num2str(gamma2)]);

subplot(3, 2, 6);
imhist(img_gamma2, 64);
title(['Histogram, Gamma = ', num2str(gamma2)]);


%% ================= Exercise 9: Histogram Equalisation =================
% Applying histogram equalization to baboon.tif
% ==================================================================
clear; close all; clc;

% --- Load and Prepare Original Image ---

% Load image
img_orig = imread('baboon.tif');

% Convert to double format
img_double = im2double(img_orig);

% --- Apply Histogram Equalization ---

num_bins_eq = 64;
img_equalized = histeq(img_double, num_bins_eq);

% --- Display Results ---

figure; % Create a new figure window

% Display original image
subplot(2, 2, 1);
imshow(img_double);
title('Original baboon.tif (double)');

% Display histogram of original image
subplot(2, 2, 2);
imhist(img_double, 64);
title('Histogram of Original (64 bins)');

% Display equalized image
subplot(2, 2, 3);
imshow(img_equalized);
title('Equalized baboon.tif');

% Display histogram of equalized image
subplot(2, 2, 4);
imhist(img_equalized, 64);
title('Histogram of Equalized (64 bins)');


%% ================= Exercise 10: Thresholding ==================
% Using a custom threshold function on medical.tif
% ============================================================
clear; close all; clc;

% --- Load and Prepare Original Image ---

% Load the image
img_orig = imread('medical.tif');

% Convert to double format
img_double = im2double(img_orig);

% --- Define Thresholds (Estimate these based on image inspection!) ---
% These are initial guesses, you may need to adjust T1 and T2
% after looking at the 'medical.tif' image or its histogram.

% Scenario a) Air (likely dark)
T1_air = 0;   T2_air = 0.2;

% Scenario b) Blood (likely mid-tones)
T1_blood = 0.4; T2_blood = 0.6;

% Scenario c) Bone (likely bright)
T1_bone = 0.8;  T2_bone = 1.0;

% --- Apply Thresholding using the function ---

img_air = my_threshold(img_double, T1_air, T2_air);
img_blood = my_threshold(img_double, T1_blood, T2_blood);
img_bone = my_threshold(img_double, T1_bone, T2_bone);

% --- Display Results ---

figure; % Create a new figure window

% Display original image
subplot(2, 2, 1);
imshow(img_double);
title('Original medical.tif (double)');

% Display thresholded image for Air
subplot(2, 2, 2);
imshow(img_air);
title(['Thresholded Air (', num2str(T1_air), '-', num2str(T2_air), ')']);

% Display thresholded image for Blood
subplot(2, 2, 3);
imshow(img_blood);
title(['Thresholded Blood (', num2str(T1_blood), '-', num2str(T2_blood), ')']);

% Display thresholded image for Bone
subplot(2, 2, 4);
imshow(img_bone);
title(['Thresholded Bone (', num2str(T1_bone), '-', num2str(T2_bone), ')']);

sgtitle('Exercise 10: Thresholding Results'); % Optional title



function out = my_threshold(I, T1, T2)
% Applies thresholding to image I.
% Pixels with values between T1 and T2 (inclusive) are set to 1, others to 0.

% Ensure input is double for comparison
I_double = im2double(I);

% Create logical mask: 1 where condition is true, 0 otherwise
mask = (I_double >= T1) & (I_double <= T2);

% Convert logical mask to double for output (0.0 or 1.0)
out = double(mask);

end