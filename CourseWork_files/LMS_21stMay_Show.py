import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Function to read a noise image in grayscale
def read_noise_image(noise_image_path):
    noise_image = cv2.imread(noise_image_path, cv2.IMREAD_GRAYSCALE)
    return noise_image

# Function to estimate local variance in an image using a specified window size
def estimate_local_variance(image, window_size=(7, 7)):
    # Pad the image to handle border pixels
    padded_image = cv2.copyMakeBorder(image, *[(s - 1) // 2 for s in window_size] * 2, cv2.BORDER_REFLECT)
    local_variance = np.zeros_like(image, dtype=np.float64)
    
    # Calculate local variance for each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            patch = padded_image[i:i+window_size[0], j:j+window_size[1]]
            local_variance[i, j] = np.var(patch)
    return local_variance

# Function to add noise to an image
def add_noise(original_image, noise_image):
    # Ensure the noise image and original image have the same dimensions
    if original_image.shape != noise_image.shape:
        raise ValueError("Noise image and original image must have the same shape.")
    
    # Normalize noise image
    noise_image = noise_image.astype(np.float64)
    noise_image = (noise_image - np.mean(noise_image)) / np.std(noise_image)
    noise_std = 25  # Adjust this value to control noise strength
    
    # Add noise to the original image
    noisy_image = original_image + noise_std * noise_image
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Function to denoise an image using the LMS algorithm
def lms_denoise(original_image, noisy_image, filter_size=(3, 3), step_size=0.01, num_iterations=10, convergence_threshold=1e-6):
    h, w = original_image.shape
    filter_height, filter_width = filter_size
    filter_weights = np.random.randn(filter_height, filter_width) * 0.01  # Small random values for initialization

    for iteration in range(num_iterations):
        predictions = np.zeros_like(noisy_image, dtype=np.float64)

        # Apply filter to predict the denoised image
        for i in range(h):
            for j in range(w):
                i_min = max(0, i - filter_height // 2)
                i_max = min(h, i + filter_height // 2 + 1)
                j_min = max(0, j - filter_width // 2)
                j_max = min(w, j + filter_width // 2 + 1)
                
                patch = noisy_image[i_min:i_max, j_min:j_max]
                if patch.shape != filter_weights.shape:
                    continue

                prediction = np.sum(patch * filter_weights)
                predictions[i, j] = prediction

        # Calculate errors and update filter weights
        errors = noisy_image - predictions
        for i in range(h):
            for j in range(w):
                i_min = max(0, i - filter_height // 2)
                i_max = min(h, i + filter_height // 2 + 1)
                j_min = max(0, j - filter_width // 2)
                j_max = min(w, j + filter_width // 2 + 1)
                
                patch = noisy_image[i_min:i_max, j_min:j_max]
                if patch.shape != filter_weights.shape:
                    continue
                
                filter_weights += step_size * errors[i, j] * patch

        # Check for convergence
        error_norm = np.linalg.norm(errors)
        if error_norm < convergence_threshold:
            break

    # Reconstruct the denoised image using the final filter weights
    return reconstruct_denoised_image(noisy_image, filter_weights, original_image.shape)

# Function to reconstruct the denoised image from the filter weights
def reconstruct_denoised_image(noisy_image, filter_weights, output_shape):
    patch_h, patch_w = filter_weights.shape
    padded_noisy_image = np.pad(noisy_image, ((patch_h // 2, patch_h // 2), (patch_w // 2, patch_w // 2)), mode='reflect')
    patches = np.lib.stride_tricks.sliding_window_view(padded_noisy_image, (patch_h, patch_w))
    denoised_image = np.tensordot(patches, filter_weights, axes=((2, 3), (0, 1)))
    denoised_image = denoised_image[:output_shape[0], :output_shape[1]]  # Ensure the output shape matches the input shape

    # Normalize the denoised image
    denoised_image = (denoised_image - denoised_image.min()) * 255 / (denoised_image.max() - denoised_image.min())
    return denoised_image

# Function to adjust the filter size to find the best denoising performance
def adjust_filter_size(original_image, noisy_image, max_filter_size=(5, 5), step_size=0.01, num_iterations=10):
    best_psnr = 0
    best_filter_size = None

    # Iterate over possible filter sizes to find the best one based on PSNR
    for filter_size_h in range(1, max_filter_size[0] + 1):
        for filter_size_w in range(1, max_filter_size[1] + 1):
            filter_size = (filter_size_h, filter_size_w)
            try:
                denoised_image = lms_denoise(original_image, noisy_image, filter_size=filter_size, step_size=step_size, num_iterations=num_iterations)
                denoised_image = np.clip(denoised_image, 0, 255).astype(original_image.dtype)
                
                psnr_value = psnr(original_image, denoised_image)
                if psnr_value > best_psnr:
                    best_psnr = psnr_value
                    best_filter_size = filter_size
            except ValueError as e:
                print(f"Skipping filter size {filter_size} due to error: {e}")

    return best_filter_size, best_psnr

# Load the original image
original_image = cv2.imread('/home/ansor/Desktop/CodesMarch/image.jpg', cv2.IMREAD_GRAYSCALE)

# Load the noise image
noise_image = read_noise_image('/home/ansor/Desktop/CodesMarch/noise_image.jpg')

# Check dimensions of noisy image and original image
print("Dimensions of original image:", original_image.shape)
print("Dimensions of noisy image:", noise_image.shape)

# Resize noise image if it doesn't match the original image dimensions
if original_image.shape != noise_image.shape:
    noise_image = cv2.resize(noise_image, (original_image.shape[1], original_image.shape[0]))
    print("Noisy image dimensions adjusted to match original image dimensions.")

# Add noise to the original image
noisy_image = add_noise(original_image, noise_image)

# Estimate local variance of the noisy image
local_variance = estimate_local_variance(noisy_image)

# Adjust filter size for the best denoising performance
adapted_filter_size, best_psnr = adjust_filter_size(original_image, noisy_image)

# Denoise the image using the adapted filter size
denoised_image = lms_denoise(original_image, noisy_image, filter_size=adapted_filter_size)

# Check and normalize intensity range of the denoised image
denoised_min = denoised_image.min()
denoised_max = denoised_image.max()

print("Intensity range of denoised image before normalization:")
print("Min:", denoised_min)
print("Max:", denoised_max)

# Check for NaN or infinite values in the denoised image
if np.isnan(denoised_min) or np.isinf(denoised_min) or np.isnan(denoised_max) or np.isinf(denoised_max):
    print("Warning: Denoised image contains NaN or infinite values. Skipping normalization.")
else:
    if denoised_min != denoised_max:
        denoised_image = (denoised_image - denoised_min) * 255 / (denoised_max - denoised_min)
        denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    else:
        print("Warning: Denoised image intensity range is zero. Skipping normalization.")

# Ensure final denoised image is within valid intensity range
denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

print("NaN or infinite values in denoised image:", np.isnan(denoised_image).any() or np.isinf(denoised_image).any())

# Print data type of the original image
print("Data type of original image:", original_image.dtype)

print("Intensity range of denoised image after normalization:")
print("Min:", denoised_image.min())
print("Max:", denoised_image.max())

print("Unique values in denoised image:", np.unique(denoised_image))

# Calculate PSNR and SSIM of the denoised image
psnr_denoised = psnr(original_image, denoised_image)
ssim_denoised = ssim(original_image, denoised_image, data_range=denoised_image.max() - denoised_image.min())

# Calculate SNR of the denoised image
signal_power = np.sum(original_image.astype(np.float64) ** 2)
noise_power = np.sum((original_image.astype(np.float64) - denoised_image.astype(np.float64)) ** 2)
snr_denoised = 10 * np.log10(signal_power / noise_power)

print("PSNR (Peak Signal-to-Noise Ratio):")
print(f"Denoised: {psnr_denoised:.2f} dB")

print("\nSSIM (Structural Similarity Index):")
print(f"Denoised: {ssim_denoised:.2f}")

print("\nSNR (Signal-to-Noise Ratio):")
print(f"Denoised: {snr_denoised:.2f} dB")

# Plot original, noisy, and denoised images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

axes[2].imshow(denoised_image, cmap='gray')
axes[2].set_title('Denoised Image')
axes[2].axis('off')

plt.show()