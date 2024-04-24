% Load image package
pkg load image

% Load original images
original_images = cell(4, 1);
for i = 1:4
    original_images{i} = imread(sprintf('/home/ansor/Downloads/CourseWorkSSimages/NLMS/image%d.jpg', i));
end

% Apply NLMS filtering to original images
nlms_images = cell(4, 1);
for i = 1:4
    % Apply NLMS filtering (replace with your NLMS filtering implementation)
    filter_size = 16; % Define the size of the filter kernel (e.g., 3x3)
    kernel = ones(filter_size) / (filter_size^2); % Averaging filter kernel (placeholder)
    nlms_images{i} = imfilter(original_images{i}, kernel, 'conv', 'replicate'); % Apply the filter
    
    % Calculate PSNR between original and manipulated images
    psnr_values(i) = psnr(original_images{i}, nlms_images{i});
    
    % Calculate SNR
    snr_values(i) = 10 * log10(mean(original_images{i}(:).^2) / mean((original_images{i}(:) - double(nlms_images{i}(:))).^2));
    
    % Calculate SSIM
    ssim_values(i) = ssim_index(original_images{i}, nlms_images{i});
    
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
    imshow(nlms_images{i});
    title(sprintf('NLMS Filtered Image %d', i));
end

% Save manipulated images
for i = 1:4
    imwrite(nlms_images{i}, sprintf('/home/ansor/Downloads/CourseWorkSSimages/NLMS/Manipulated/manipulated_image%d.jpg', i));
end
