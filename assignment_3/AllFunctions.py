import numpy as np
import scipy.ndimage as ndi
import scipy.fft as fft
import skimage.io as io
import skimage.filters as flt
import math
import matplotlib.pyplot as plt


def main():
    """The code for displaying the results"""

    noisy = io.imread('./noisy.tif')

    # Question 1 (a)
    denoised_1 = square_average_filter('./noisy.tif', 5)
    denoised_2 = square_average_filter('./noisy.tif', 10)
    denoised_3 = square_average_filter('./noisy.tif', 15)
    fig, plots = plt.subplots(1, 4)
    fig.suptitle('Question 1 (a):Average filter')
    plots[0].imshow(noisy, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(denoised_1, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title('Size = 5')
    plots[2].imshow(denoised_2, cmap='gray', vmax=255, vmin=0)
    plots[2].set_title('Size = 10')
    plots[3].imshow(denoised_3, cmap='gray', vmin=0, vmax=255)
    plots[3].set_title('Size = 15')
    plt.show()

    denoised, sharpened = high_boost_filter('./noisy.tif')
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 1 (b):Sharpening')
    plots[0].imshow(noisy, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(denoised, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title('Denoised image')
    plots[2].imshow(sharpened, cmap='gray', vmax=255, vmin=0)
    plots[2].set_title('Sharpened image')
    plt.show()

    # Question 2(a)
    sinusoidal_image = generate_sinusoidal_image(1001, 1001)
    _, dft_spectrum = dft(sinusoidal_image)
    fig, plots = plt.subplots(1, 2)
    fig.suptitle('Question 2 (a):DFT')
    plots[0].imshow(sinusoidal_image, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Sinusoidal image')
    plots[1].imshow(dft_spectrum, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title('DFT spectrum')
    plt.show()

    char_image = io.imread('./characters.tif')
    # Question 2(b)
    dft_data, dft_spectrum = dft(char_image)
    filtered_dft, filtered_spectrum = filter_ideal_low_pass(dft_data, 100)
    filtered_image = inverse_dft(filtered_dft)
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 2 (b):Ideal low pass filtering')
    plots[0, 0].imshow(char_image, cmap='gray')
    plots[0, 0].set_title('Original image')
    plots[0, 1].imshow(dft_spectrum, cmap='gray')
    plots[0, 1].set_title('Centered DFT spectrum')
    plots[1, 0].imshow(filtered_spectrum, cmap='gray')
    plots[1, 0].set_title('Low pass filtered Spectrum')
    plots[1, 1].imshow(filtered_image, cmap='gray')
    plots[1, 1].set_title('Filtered Image')
    plt.show()

    # Question 2(c)
    gauss_filt_dft, gauss_filt_spectrum = filter_gaussian_low_pass(
        dft_data, 100)
    gaussian_filtered_image = inverse_dft(gauss_filt_dft)
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 2 (b):Gaussian low pass filtering')
    plots[0, 0].imshow(char_image, cmap='gray')
    plots[0, 0].set_title('Original image')
    plots[0, 1].imshow(dft_spectrum, cmap='gray')
    plots[0, 1].set_title('Centered DFT spectrum')
    plots[1, 0].imshow(gauss_filt_spectrum, cmap='gray')
    plots[1, 0].set_title('Low pass filtered Spectrum')
    plots[1, 1].imshow(gaussian_filtered_image, cmap='gray')
    plots[1, 1].set_title('Filtered Image')
    plt.show()

    # comparing ideal and gaussian lpf results
    fig, plots = plt.subplots(1, 2)
    fig.suptitle('Comparing Ideal vs Gaussian LPF')
    plots[0].imshow(filtered_image, cmap='gray')
    plots[0].set_title('Result of Ideal LPF (D_0 = 100)')
    plots[1].imshow(gaussian_filtered_image, cmap='gray')
    plots[1].set_title('Result of Gaussian LPF (D_0 = 100)')
    plt.show()

    # Question 3
    pet_image = io.imread('./PET_image.tif').astype(np.int64)
    spectrum, fil_spectrum, fil_image = homomorphic_filter('./PET_image.tif')
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('PET Image HPF')
    plots[0, 0].imshow(pet_image, cmap='gray', vmin=0, vmax=255)
    plots[0, 0].set_title('Original Image')
    plots[0, 1].imshow(spectrum, cmap='gray')
    plots[0, 1].set_title('Centered DFT spectrum')
    plots[1, 0].imshow(fil_spectrum, cmap='gray')
    plots[1, 0].set_title('High pass filtered DFT spectrum')
    plots[1, 1].imshow(fil_image, cmap='gray')
    plots[1, 1].set_title('Filtered Image')
    plt.show()


def square_average_filter(image_path, size):
    image_data = io.imread(image_path)
    # convolve with averaging kernel of required size
    kernel = np.ones((size, size)) / (size**2)
    denoised_image = ndi.convolve(image_data, kernel, mode='reflect')
    return denoised_image


def high_boost_filter(image_path):
    # the denoised and then blurred images
    reference_image = io.imread('./characters.tif')
    denoised_image = square_average_filter('./noisy.tif', 5)
    blurred_image = flt.gaussian(denoised_image, 3) * 255

    # the unsharp mask
    mask = (denoised_image - blurred_image).astype(np.int64)

    # optimizing k value for mse
    k_values = np.arange(-10, 10, 0.5)
    errors = np.zeros_like(k_values)
    for i in range(len(k_values)):
        sharpened_image = denoised_image + k_values[i] * mask
        error = np.square(sharpened_image - reference_image).mean()
        errors[i] = error
    k_optimum = k_values[np.argmin(errors)]

    # applying the sharpening
    filtered_image = denoised_image + k_optimum * mask
    filtered_image = filtered_image.clip(0, 255)
    return denoised_image, filtered_image


def generate_sinusoidal_image(M, N):
    sin_image = np.zeros((M, N))
    rows, columns = sin_image.shape
    u = 100
    v = 200
    for m in range(rows):
        for n in range(columns):
            sin_image[m, n] = math.sin(2 * math.pi * (u * m / M + v * n / N))
    sin_image = (sin_image + 1) / 2
    return sin_image * 255


def dft(image):
    dft = fft.fft2(image)
    dft_centered = fft.fftshift(dft)
    dft_spectrum = np.log(1 + np.abs(dft_centered)) * 255
    return dft_centered, dft_spectrum


def inverse_dft(dft_data):
    fft_decentralized = fft.ifftshift(dft_data)
    idft = fft.ifft2(fft_decentralized)
    filtered_image = np.abs(idft) * 255
    return filtered_image


def filter_ideal_low_pass(dft_centered, D_0=100):
    H = np.zeros_like(dft_centered, dtype=np.float64)
    P, Q = dft_centered.shape
    # calculate the low pass filter
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            if D > D_0:
                H[u, v] = 0
            else:
                H[u, v] = 1
    # apply the filter
    filtered_dft = H * dft_centered
    filtered_spectrum = H * np.log(1 + np.abs(filtered_dft)) * 255
    return filtered_dft, filtered_spectrum


def filter_gaussian_low_pass(dft_centered, D_0=100):
    H = np.zeros_like(dft_centered, dtype=np.float64)
    P, Q = dft_centered.shape
    # calculate the gaussian low pass filter
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            H[u, v] = math.exp((-D**2) / (2 * (D_0**2)))
    filtered_dft = H * dft_centered
    filtered_spectrum = H * np.log(1 + np.abs(filtered_dft)) * 255
    return filtered_dft, filtered_spectrum


def homomorphic_filter(image_path):
    image_data = io.imread(image_path).astype(np.int64)
    image_log = np.log(1 + image_data)
    dft_data, dft_spectrum = dft(image_log)
    filtered_dft, filtered_spectrum = filter_high_pass(
        dft_data, 100, 2.2, 0.25)
    filtered_image = inverse_dft(filtered_dft)
    return dft_spectrum, filtered_spectrum, filtered_image


def filter_high_pass(dft_centered, D_0=100, gammaH=2.2, gammaL=0.25):
    H = np.zeros_like(dft_centered, dtype=np.float64)
    P, Q = dft_centered.shape
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            H[u, v] = (gammaH - gammaL) * \
                (1 - math.exp((-D**2) / (2 * (D_0**2)))) + gammaL
    filtered_dft = H * dft_centered
    filtered_spectrum = H * np.log(1 + np.abs(filtered_dft)) * 255
    return filtered_dft, filtered_spectrum


def full_scale_contrast_stretch(image_data):
    max_val = image_data.max()
    min_val = image_data.min()
    stretched_image = ((image_data - min_val) / (max_val - min_val)) * 255
    return stretched_image


if __name__ == "__main__":
    main()
