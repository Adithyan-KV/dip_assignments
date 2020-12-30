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

    # # Question 1
    # filtered = inverse_filter(low_noise, kernel)

    # Question 2
    denoised = gaussian_denoise(noisy, 7, 3)
    plt.imshow(noisy, cmap='gray')
    plt.figure()
    plt.imshow(denoised, cmap='gray')
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


if __name__ == "__main__":
    main()
