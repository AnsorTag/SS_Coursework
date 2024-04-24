function ssim_value = ssim_index(img1, img2)
    % Compute mean of images
    mu1 = mean(img1(:));
    mu2 = mean(img2(:));

    % Compute variance of images
    sigma1_squared = var(img1(:));
    sigma2_squared = var(img2(:));

    % Compute covariance
    covariance = cov(img1(:), img2(:));

    % Constants
    c1 = (0.01 * 255)^2;
    c2 = (0.03 * 255)^2;

    % Compute SSIM index
    numerator = (2 * mu1 * mu2 + c1) * (2 * covariance + c2);
    denominator = (mu1^2 + mu2^2 + c1) * (sigma1_squared + sigma2_squared + c2);
    ssim_value = mean(numerator(:)) / mean(denominator(:));
end
