% Load image package
pkg load image

% Load original images
original_images = cell(4, 1);
for i = 1:4
    original_images{i} = imread(sprintf('/home/ansor/Downloads/CourseWorkSSimages/LMS/image%d.jpg', i));
end

% Apply LMS filtering to original images
lms_images = cell(4, 1);

% Define PSNR, SNR, and SSIM arrays
psnr_values = zeros(4, 1);
snr_values = zeros(4, 1);
ssim_values = zeros(4, 1);

for i = 1:4
    % Apply LMS filtering
    % Replace the following line with your LMS filtering implementation
    filter_size = 36; % Define the size of the filter kernel (e.g., 3x3)
    kernel = ones(filter_size) / (filter_size^2); % Averaging filter kernel
    lms_images{i} = imfilter(original_images{i}, kernel, 'conv', 'replicate');
    
    % Calculate PSNR between original and manipulated images
    psnr_values(i) = psnr(original_images{i}, lms_images{i});
    
    % Calculate SNR
    snr_values(i) = 10 * log10(mean(original_images{i}(:).^2) / mean((original_images{i}(:) - double(lms_images{i}(:))).^2));
    
    % Calculate SSIM
    ssim_values(i) = ssim_index(original_images{i}, lms_images{i});
    
    % Display PSNR, SNR, and SSIM values
    fprintf('Image %d: PSNR = %.2f dB, SNR = %.2f dB, SSIM = %.4f\n', i, psnr_values(i), snr_values(i), ssim_values(i));
end

% Display images pair by pair in separate figures
for i = 1:4
    figure;
    subplot(1, 2, 1);
    imshow(original_images{i});
    title(sprintf('Original Image %d', i));
    subplot(1, 2, 2);
    imshow(lms_images{i});
    title(sprintf('LMS Filtered Image %d', i));
end

% Save manipulated images
for i = 1:4
    imwrite(lms_images{i}, sprintf('/home/ansor/Downloads/CourseWorkSSimages/LMS/Manipulated/image%d_lms.jpg', i));
end
