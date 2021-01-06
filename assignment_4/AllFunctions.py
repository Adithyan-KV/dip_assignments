import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fft as fft
import scipy.ndimage as ndi
import skimage.io as io
from scipy.io import loadmat
import time


def main():
    kernel_obj = loadmat('BlurKernel.mat')
    kernel = kernel_obj['h']
    low_noise = io.imread('Blurred-LowNoise.png')
    med_noise = io.imread('Blurred-MedNoise.png')
    high_noise = io.imread('Blurred-HighNoise.png')
    noisy = io.imread('noisy-book1.png')
    noisy_2 = io.imread('noisy-book2.png')
    barbara = io.imread('barbara.tif')

    # # Question 1
    # filtered = inverse_filter(low_noise, kernel)

    # # Question 2
    # denoised = gaussian_denoise(noisy, 7, 3)
    # plt.imshow(noisy, cmap='gray')
    # plt.figure()
    # plt.imshow(denoised, cmap='gray')
    # plt.show()

    # denoised_median = median_filter_denoise(noisy, 5)
    # plt.imshow(noisy, cmap='gray')
    # plt.figure()
    # plt.imshow(denoised_median, cmap='gray')
    # plt.show()

    # denoised_bilateral = bilateral_filter(noisy_2, 3, 2, 2)
    # plt.imshow(noisy_2, cmap='gray')
    # plt.figure()
    # plt.imshow(denoised_bilateral, cmap='gray')
    # plt.show()

    downsampled_barb = downsample(barbara, 2)
    plt.imshow(downsampled_barb, cmap='gray')
    plt.figure()
    decimated_barb = decimate(barbara, 2)
    plt.imshow(decimated_barb, cmap='gray')
    plt.show()


def inverse_filter(image_data, kernel):
    image_dft, image_spectrum = dft(image_data)

    padded_kernel = pad_to_be_like(kernel, image_data)

    kernel_dft, kernel_spectrum = dft(padded_kernel)
    plt.imshow(image_spectrum, cmap='gray')

    original_image_dft = image_dft * 1 / kernel_dft
    original_image_spectrum = np.log(1 + np.abs(original_image_dft))
    original_image = inverse_dft(original_image_dft)
    plt.figure()
    plt.imshow(original_image_spectrum, cmap='gray')

    plt.figure()
    plt.imshow(kernel_spectrum, cmap='gray')
    plt.figure()
    plt.imshow(original_image, cmap='gray')
    plt.show()


def pad_to_be_like(kernel, image):
    rows_k, cols_k = kernel.shape
    rows_i, cols_i = image.shape
    pad_h = cols_i - cols_k
    pad_v = rows_i - rows_k

    pad_left = int(pad_h / 2)
    pad_right = cols_i - cols_k - pad_left
    pad_top = int(pad_v / 2)
    pad_bottom = rows_i - rows_k - pad_top
    padded_kernel = np.pad(
        kernel, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant')
    return padded_kernel


def dft(image_data):
    dft_image = fft.fft2(image_data)
    dft_centered = fft.fftshift(dft_image)
    dft_spectrum = np.log(1 + np.abs(dft_centered))
    return dft_centered, dft_spectrum


def inverse_dft(dft_data):
    dft_decentralized = fft.ifftshift(dft_data)
    idft = fft.ifft2(dft_decentralized)
    filtered_image = np.abs(idft)
    return filtered_image


def gaussian_denoise(image_data, kernel_size, std):
    kernel = generate_gaussian_kernel(kernel_size, std)
    denoised_image = ndi.convolve(image_data, kernel)
    return denoised_image


def median_filter_denoise(image_data, kernel_size):
    padding = int(kernel_size / 2)
    padded_image = np.pad(image_data, padding, 'reflect')
    filtered_image = np.zeros_like(image_data)
    rows, columns = padded_image.shape
    for i in range(padding, rows - padding):
        for j in range(padding, columns - padding):
            window = padded_image[i - padding:i +
                                  padding + 1, j - padding:j + padding + 1]
            filtered_image[i - padding, j - padding] = np.median(window)
    return filtered_image


def bilateral_filter(image_data, kernel_size, std_dist, std_lum):
    distance_kernel = generate_gaussian_kernel(kernel_size, std_dist)
    padding = int(kernel_size / 2)
    padded_image = np.pad(image_data, padding, 'reflect')
    filtered_image = np.zeros_like(image_data)
    rows, columns = padded_image.shape
    for i in range(padding, rows - padding):
        for j in range(padding, columns - padding):
            window = padded_image[i - padding:i +
                                  padding + 1, j - padding:j + padding + 1]
            value = padded_image[i, j]
            lum_kernel = np.square(window - value)
            sum_kernel = lum_kernel.sum()
            if sum_kernel > 0:
                lum_kernel = lum_kernel / lum_kernel.sum()
            filtered_image[i - padding, j - padding] = (
                lum_kernel * distance_kernel * window).sum()
    return filtered_image


def generate_gaussian_kernel(size, std):
    # shifting coordinates to center of kernel
    mid = int(size / 2)

    # generating kernel
    x_range = np.arange(-mid, mid + 1, 1)
    y_range = np.arange(-mid, mid + 1, 1)
    x, y = np.meshgrid(x_range, y_range)
    kernel = np.exp((np.square(x) + np.square(y)) / (2 * std**2))

    # normalizing kernel
    normalized_kernel = kernel / kernel.sum()

    return normalized_kernel


def downsample(image_data, factor):
    downsampled_image = image_data[0::factor, 0::factor]
    return downsampled_image


def decimate(image_data, factor):
    dft_data, _ = dft(image_data)
    dft_filtered, _ = filter_gaussian_low_pass(dft_data)
    filtered_image = inverse_dft(dft_filtered)
    decimated = downsample(filtered_image, factor)
    return decimated


def filter_gaussian_low_pass(dft_centered, D_0=100):
    H = np.zeros_like(dft_centered, dtype=np.float64)
    P, Q = dft_centered.shape
    # calculate the gaussian low pass filter
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            H[u, v] = math.exp((-D**2) / (2 * (D_0**2)))
    filtered_dft = H * dft_centered
    filtered_spectrum = np.log(1 + np.abs(filtered_dft)) * 255
    return filtered_dft, filtered_spectrum


if __name__ == "__main__":
    main()
