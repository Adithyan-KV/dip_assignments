import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fft as fft
import scipy.ndimage as ndi
from scipy.stats import spearmanr
import skimage.io as io
import skimage.color as col
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from scipy.io import loadmat
import time


def main():
    # kernel_obj = loadmat('BlurKernel.mat')
    # kernel = kernel_obj['h']
    # low_noise = io.imread('Blurred-LowNoise.png')
    # med_noise = io.imread('Blurred-MedNoise.png')
    # high_noise = io.imread('Blurred-HighNoise.png')
    # original_book = io.imread('Original-book.png')
    # noisy = io.imread('noisy-book1.png')
    # noisy_2 = io.imread('noisy-book2.png')
    # barbara = io.imread('barbara.tif')
    # donut = io.imread('donut.jpg')
    # phone = io.imread('phone.jpg')
    # dude = io.imread('dude.jpg')

    ref_obj = loadmat('hw5.mat')
    ref_image_names = ref_obj['refnames_blur']
    human_opinion_scores = ref_obj['blur_dmos'][0]
    blur_orgs = ref_obj['blur_orgs']
    ignore_elem_number = blur_orgs.sum()
    mse_list = get_mses(ref_image_names)
    ssim_list = get_ssims(ref_image_names)
    mse_list = mse_list[:-ignore_elem_number]
    ssim_list = ssim_list[:-ignore_elem_number]
    human_opinion_scores = ssim_list[:-ignore_elem_number]
    srocc_mse = spearmanr(mse_list, human_opinion_scores)
    srocc_ssim = spearmanr(ssim_list, human_opinion_scores)
    print(srocc_mse, srocc_ssim)


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


def detect_edges(image_data, size_kernel, std_kernel, threshold):
    # converting color images to grayscale
    grayscale_image = col.rgb2gray(image_data)

    # blurring the image
    blurred_image = gaussian_denoise(grayscale_image, size_kernel, std_kernel)

    # filtering using sobel kernel
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_kernel_y = np.transpose(sobel_kernel_x)
    x_magnitudes = ndi.convolve(blurred_image, sobel_kernel_x)
    y_magnitudes = ndi.convolve(blurred_image, sobel_kernel_y)
    magnitude_map = np.sqrt(np.square(x_magnitudes) + np.square(y_magnitudes))

    # thresholding edges
    edge_image = (magnitude_map > threshold)
    return edge_image


def get_mses(ref_image_names):
    num_images = len(ref_image_names[0])
    mse_list = np.zeros(num_images)
    for i in range(num_images):
        # loading in images from relative paths
        distorted_image = io.imread(f'hw5/gblur/img{i+1}.bmp')
        reference_image = io.imread(f'hw5/refimgs/{ref_image_names[0, i][0]}')

        # converting to grayscale
        dist_img_gray = col.rgb2gray(distorted_image)
        ref_img_gray = col.rgb2gray(reference_image)

        # computing MSE
        mse = mean_squared_error(dist_img_gray, ref_img_gray)
        mse_list[i] = mse
    return mse_list


def get_ssims(ref_image_names):
    num_images = len(ref_image_names[0])
    ssim_list = np.zeros(num_images)
    for i in range(num_images):
        # loading in images from relative paths
        distorted_image = io.imread(f'hw5/gblur/img{i+1}.bmp')
        reference_image = io.imread(f'hw5/refimgs/{ref_image_names[0, i][0]}')

        # converting to grayscale
        dist_img_gray = col.rgb2gray(distorted_image)
        ref_img_gray = col.rgb2gray(reference_image)

        # computing ssims
        ssim = structural_similarity(ref_img_gray, dist_img_gray)
        ssim_list[i] = ssim
    return ssim_list


if __name__ == "__main__":
    main()
