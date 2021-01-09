import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fft as fft
import scipy.ndimage as ndi
from scipy.stats import spearmanr
import skimage.io as io
import skimage.color as col
from skimage.metrics import structural_similarity, mean_squared_error
from skimage.restoration import denoise_bilateral
from scipy.io import loadmat
import time


def main():
    kernel_obj = loadmat('BlurKernel.mat')
    kernel = kernel_obj['h']
    kernel = kernel / kernel.sum()
    low_noise = io.imread('Blurred-LowNoise.png')
    # med_noise = io.imread('Blurred-MedNoise.png')
    # high_noise = io.imread('Blurred-HighNoise.png')
    original_book = io.imread('Original-book.png')
    # noisy = io.imread('noisy-book1.png')
    # noisy_2 = io.imread('noisy-book2.png')
    # barbara = io.imread('barbara.tif')
    # donut = io.imread('donut.jpg')
    # phone = io.imread('phone.jpg')
    # dude = io.imread('dude.jpg')
    # blurred = bleh(original_book, kernel)
    # blu, ble, bla, deblurred = inverse_filter(low_noise, kernel)
    # fig, plots = plt.subplots(2, 2)
    # fig.suptitle('Question 1(a):Inverse filtering')
    # plots[0, 0].imshow(blu, cmap='gray')
    # plots[0, 0].set_title('Noisy image spectrum (low noise)')
    # plots[0, 1].imshow(ble, cmap='gray')
    # plots[0, 1].set_title('Kernel Spectrum')
    # plots[1, 0].imshow(bla, cmap='gray')
    # plots[1, 0].set_title('Filtered Spectrum')
    # plots[1, 1].imshow(deblurred, cmap='gray')
    # plots[1, 1].set_title('Filtered Image')
    # plt.show()
    # wiener = wiener_filter(blurred, kernel, 0)


def inverse_filter(image_data, kernel):
    image_data = image_data.astype(np.float64) / 255
    image_dft, image_spectrum = dft(image_data)

    padded_kernel = pad_to_be_like(kernel, image_data)

    kernel_dft, kernel_spectrum = dft(padded_kernel)
    original_image_dft = np.real(image_dft * 1 / kernel_dft)

    original_image_spectrum = np.log(1 + np.abs(original_image_dft))
    original_image = inverse_dft(original_image_dft)
    return image_spectrum, kernel_spectrum, original_image_spectrum, original_image


def wiener_filter(image_data, kernel, sigma):
    image_data = image_data.astype(np.float64) / 255

    # computing the dfts
    image_dft, _ = dft(image_data)
    padded_kernel = pad_to_be_like(kernel, image_data)
    kernel_dft, _ = dft(padded_kernel)

    Sw = sigma**2
    Sf = ((np.abs(image_dft)))**2

    # computing the wiener filter
    wiener_numerator = np.conjugate(kernel_dft)
    wiener_denominator = ((np.abs(kernel_dft))**2) + (Sw / Sf)
    wiener_filter = wiener_numerator / wiener_denominator

    # applying the filter
    filtered_image_dft = image_dft * wiener_filter
    filtered_image = inverse_dft(filtered_image_dft)

    return filtered_image


def clsf(image_data, kernel, lam):
    # the laplacian kernel
    p = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]])

    # calculating DFTs
    image_dft, _ = dft(image_data)
    padded_p = pad_to_be_like(p, image_dft)
    p_dft, _ = dft(padded_p)
    padded_kernel = pad_to_be_like(kernel, image_dft)
    kernel_dft, _ = dft(padded_kernel)

    # computing the filter
    filt_numerator = np.conjugate(kernel_dft)
    filt_denominator = ((np.abs(kernel_dft))**2) + lam * (np.abs(p_dft))**2
    filt = filt_numerator / filt_denominator

    # Applying filter and restoring image
    filtered_image_dft = image_dft * filt
    filtered_image = inverse_dft(filtered_image_dft)
    return filtered_image


def pad_to_be_like(kernel, image):
    rows_k, cols_k = kernel.shape
    rows_i, cols_i = image.shape
    pad_h = cols_i - cols_k
    pad_v = rows_i - rows_k

    padded_kernel = np.pad(
        kernel, ((0, pad_v), (0, pad_h)), 'constant')
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
                                  padding + 1, j - padding: j + padding + 1]
            filtered_image[i - padding, j - padding] = np.median(window)
    return filtered_image


def bilateral_filter(image_data, kernel_size, std_dist, std_lum):
    image_data = image_data / 255
    distance_kernel = generate_gaussian_kernel(kernel_size, std_dist)
    # padding for convolution
    padding = int(kernel_size / 2)
    padded_image = np.pad(image_data, padding, 'reflect')
    filtered_image = np.zeros_like(image_data)
    rows, columns = padded_image.shape
    for i in range(padding, rows - padding):
        for j in range(padding, columns - padding):
            window = padded_image[i - padding: i +
                                  padding + 1, j - padding: j + padding + 1]
            # computing the luminance kernel
            value = padded_image[i, j]
            x = (window - value)**2
            lum_kernel = np.exp(-x / (2 * std_lum**2))

            # applying the final kernel
            final_kernel = lum_kernel * distance_kernel
            if final_kernel.sum() > 0:
                final_kernel = final_kernel / final_kernel.sum()
            filtered_image[i - padding, j -
                           padding] = (final_kernel * window).sum()
    return filtered_image


def generate_gaussian_kernel(size, std):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    x_grid, y_grid = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(x_grid) + np.square(y_grid)) / std**2)
    # normalizing kernel
    kernel = kernel / kernel.sum()

    return kernel


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


def get_sroccs():
    ref_obj = loadmat('hw5.mat')
    ref_image_names = ref_obj['refnames_blur']
    human_opinion_scores = ref_obj['blur_dmos'][0]
    # computing the metrics
    mse_list = get_mses(ref_image_names)
    ssim_list = get_ssims(ref_image_names)
    # ignoring the images which are not distorted
    mse_list = mse_list[:-29]
    ssim_list = ssim_list[:-29]
    human_opinion_scores = human_opinion_scores[:-29]
    srocc_mse = spearmanr(mse_list, human_opinion_scores)
    srocc_ssim = spearmanr(ssim_list, human_opinion_scores)
    return srocc_mse, srocc_ssim


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
