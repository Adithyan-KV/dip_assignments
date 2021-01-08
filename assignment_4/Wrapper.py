import AllFunctions as af
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.transform as tfm


def main():
    question_1()
    # question_2()
    # question_3()
    # question_4()
    # question_5()


def question_1():
    kernel_obj = loadmat('BlurKernel.mat')
    kernel = kernel_obj['h']
    plt.imshow(kernel)
    plt.show()
    low_noise = io.imread('Blurred-LowNoise.png')
    med_noise = io.imread('Blurred-MedNoise.png')
    high_noise = io.imread('Blurred-HighNoise.png')
    img_spec_1, kernel_spec_1, final_spec_1, image_1 = af.inverse_filter(
        low_noise, kernel)
    img_spec_2, kernel_spec_2, final_spec_2, image_2 = af.inverse_filter(
        med_noise, kernel)
    img_spec_3, kernel_spec_3, final_spec_3, image_3 = af.inverse_filter(
        high_noise, kernel)
    # low noise
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 1(a):Inverse filtering')
    plots[0, 0].imshow(img_spec_1, cmap='gray')
    plots[0, 0].set_title('Noisy image spectrum (low noise)')
    plots[0, 1].imshow(kernel_spec_1, cmap='gray')
    plots[0, 1].set_title('Kernel Spectrum')
    plots[1, 0].imshow(final_spec_1, cmap='gray')
    plots[1, 0].set_title('Filtered Spectrum')
    plots[1, 1].imshow(image_1, cmap='gray')
    plots[1, 1].set_title('Filtered Image')
    # # med noise
    # fig_2, plots_2 = plt.subplots(2, 2)
    # fig_2.suptitle('Question 1(a):Inverse filtering')
    # plots_2[0, 0].imshow(img_spec_2, cmap='gray')
    # plots_2[0, 0].set_title('Noisy image spectrum (medium noise)')
    # plots_2[0, 1].imshow(kernel_spec_2, cmap='gray')
    # plots_2[0, 1].set_title('Kernel Spectrum')
    # plots_2[1, 0].imshow(final_spec_2, cmap='gray')
    # plots_2[1, 0].set_title('Filtered Spectrum')
    # plots_2[1, 1].imshow(image_2, cmap='gray')
    # plots_2[1, 1].set_title('Filtered Image')
    # # high noise
    # fig_3, plots_3 = plt.subplots(2, 2)
    # fig_3.suptitle('Question 1(a):Inverse filtering')
    # plots_3[0, 0].imshow(img_spec_3, cmap='gray')
    # plots_3[0, 0].set_title('Noisy image spectrum (High noise)')
    # plots_3[0, 1].imshow(kernel_spec_3, cmap='gray')
    # plots_3[0, 1].set_title('Kernel Spectrum')
    # plots_3[1, 0].imshow(final_spec_3, cmap='gray')
    # plots_3[1, 0].set_title('Filtered Spectrum')
    # plots_3[1, 1].imshow(image_3, cmap='gray')
    # plots_3[1, 1].set_title('Filtered Image')
    # plt.show()


def question_2():
    # part a
    noisy = io.imread('noisy-book1.png')
    # gauss_denoised = af.gaussian_denoise(noisy, 7, 3)
    # median_denoised = af.median_filter_denoise(noisy, 5)
    # fig, plots = plt.subplots(1, 3)
    # fig.suptitle('Question 2 (a):Comparing median and gaussian filtering')
    # plots[0].imshow(noisy, cmap='gray')
    # plots[0].set_title('Original Image')
    # plots[1].imshow(gauss_denoised, cmap='gray')
    # plots[1].set_title('Gaussian filtered')
    # plots[2].imshow(median_denoised, cmap='gray')
    # plots[2].set_title('Median filtered')
    # plt.show()

    # part b
    noisy_2 = io.imread('noisy-book2.png')
    gauss_denoised = af.gaussian_denoise(noisy_2, 9, 6)
    bilateral_filtered = af.bilateral_filter(noisy_2, 9, 6, 0.5)
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 2 (a):Comparing median and gaussian filtering')
    plots[0].imshow(noisy_2, cmap='gray')
    plots[0].set_title('Original Image')
    plots[1].imshow(gauss_denoised, cmap='gray')
    plots[1].set_title('Gaussian filtered')
    plots[2].imshow(bilateral_filtered, cmap='gray')
    plots[2].set_title('Bilateral filtered')
    plt.show()


def question_3():
    barbara = io.imread("barbara.tif")
    downsampled = af.downsample(barbara, 2)
    decimated = af.decimate(barbara, 2)
    scaled_using_library = tfm.rescale(barbara, 1 / 2)
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 3:Downsampling and Decimating')
    plots[0, 0].imshow(barbara, cmap='gray')
    plots[0, 0].set_title('Original Image')
    plots[0, 1].imshow(downsampled, cmap='gray')
    plots[0, 1].set_title('Downsampled by 2x')
    plots[1, 0].imshow(decimated, cmap='gray')
    plots[1, 0].set_title('Decimated by 2x')
    plots[1, 1].imshow(scaled_using_library, cmap='gray')
    plots[1, 1].set_title('Using skimage library function')
    plt.show()


def question_4():
    # All free stock images taken from Unsplash.com
    donut = io.imread('donut.jpg')
    phone = io.imread('phone.jpg')
    dude = io.imread('dude.jpg')
    street = io.imread('street.jpg')
    donut_edges = af.detect_edges(donut, 5, 3, 0.5)
    phone_edges = af.detect_edges(phone, 7, 5, 0.2)
    dude_edges = af.detect_edges(dude, 5, 3, 0.15)
    street_edges = af.detect_edges(street, 5, 3, 0.4)
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 4:Edge detection')
    plots[0, 0].imshow(donut)
    plots[0, 0].set_title('Original Image')
    plots[0, 1].imshow(donut_edges, cmap='gray')
    plots[0, 1].set_title('Detected Edges')
    plots[1, 0].imshow(phone)
    plots[1, 0].set_title('Original Image')
    plots[1, 1].imshow(phone_edges, cmap='gray')
    plots[1, 1].set_title('Detected Edges')
    fig_2, plots_2 = plt.subplots(2, 2)
    fig_2.suptitle('Question 4:Edge detection')
    plots_2[0, 0].imshow(dude)
    plots_2[0, 0].set_title('Original Image')
    plots_2[0, 1].imshow(dude_edges, cmap='gray')
    plots_2[0, 1].set_title('Detected Edges')
    plots_2[1, 0].imshow(street)
    plots_2[1, 0].set_title('Original Image')
    plots_2[1, 1].imshow(street_edges, cmap='gray')
    plots_2[1, 1].set_title('Detected Edges')
    plt.show()


def question_5():
    srocc_mse, srocc_ssim = af.get_sroccs()
    print(f"for MSE metric: {srocc_mse}")
    print(f"for SSIM metric:{srocc_ssim}")


if __name__ == "__main__":
    main()
