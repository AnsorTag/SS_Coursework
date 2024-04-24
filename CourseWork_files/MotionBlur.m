% Load image package
pkg load image

% Load original images
original_images = cell(4, 1);
for i = 1:4
    original_images{i} = imread(sprintf('/home/ansor/Downloads/CourseWorkSSimages/MotionBlur/image%d.jpg', i));
end

% Apply Motion Blur to original images
motion_blur_images = cell(4, 1);
for i = 1:4
    % Apply Motion Blur
    PSF = fspecial('motion', 20, 45); % Adjust parameters as needed
    motion_blur_images{i} = imfilter(original_images{i}, PSF, 'conv', 'circular');
    
    % Calculate PSNR between original and manipulated images
    psnr_value = psnr(original_images{i}, motion_blur_images{i});
    
    % Calculate SNR
    snr_value = 10 * log10(mean(original_images{i}(:).^2) / mean((original_images{i}(:) - double(motion_blur_images{i}(:))).^2));
    
    % Calculate SSIM
    ssim_value = ssim_index(original_images{i}, motion_blur_images{i});
    
    % Display and save PSNR, SNR, and SSIM values
    fprintf('Image %d: PSNR = %.2f dB, SNR = %.2f dB, SSIM = %.4f\n', i, psnr_value, snr_value, ssim_value);
end

% Display images pair by pair in separate figures
for i = 1:4
    figure;
    subplot(1, 2, 1);
    imshow(original_images{i});
    title(sprintf('Original Image %d', i));
    subplot(1, 2, 2);
    imshow(motion_blur_images{i});
    title(sprintf('Motion Blur Image %d', i));
end

% Save manipulated images
for i = 1:4
    imwrite(motion_blur_images{i}, sprintf('/home/ansor/Downloads/CourseWorkSSimages/MotionBlur/Manipulated/manipulated_image%d.jpg', i));
end
