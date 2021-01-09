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
    low_noise = io.imread('Blurred-LowNoise.png')
    med_noise = io.imread('Blurred-MedNoise.png')
    high_noise = io.imread('Blurred-HighNoise.png')
    img_spec_1, kernel_spec_1, final_spec_1, image_1 = af.inverse_filter(
        low_noise, kernel)
    img_spec_2, kernel_spec_2, final_spec_2, image_2 = af.inverse_filter(
        med_noise, kernel)
    img_spec_3, kernel_spec_3, final_spec_3, image_3 = af.inverse_filter(
        high_noise, kernel)
    fig, plots = plt.subplots(3, 5)
    fig.suptitle('Question 1 (c):Constrained Least Squares Filtering')
    plots[0, 0].imshow(low_noise, cmap='gray')
    plots[0, 0].set_title('Original Image- Low noise Blurred')
    plots[0, 1].imshow(img_spec_1, cmap='gray')
    plots[0, 1].set_title('Image spectrum-Low noise')
    plots[0, 2].imshow(kernel_spec_1, cmap='gray')
    plots[0, 2].set_title('kernel spectrum-Low noise')
    plots[0, 3].imshow(final_spec_1, cmap='gray')
    plots[0, 3].set_title('Filtered spectrum-Low noise')
    plots[0, 4].imshow(image_1, cmap='gray')
    plots[0, 4].set_title('Restored Image-Low noise')
    plots[1, 0].imshow(med_noise, cmap='gray')
    plots[1, 0].set_title('Original Image- Medium noise Blurred')
    plots[1, 1].imshow(img_spec_2, cmap='gray')
    plots[1, 1].set_title('Image spectrum-Medium noise')
    plots[1, 2].imshow(kernel_spec_2, cmap='gray')
    plots[1, 2].set_title('kernel spectrum-Medium noise')
    plots[1, 3].imshow(final_spec_2, cmap='gray')
    plots[1, 3].set_title('Filtered spectrum-Medium noise')
    plots[1, 4].imshow(image_2, cmap='gray')
    plots[1, 4].set_title('Restored Image-Medium noise')
    plots[2, 0].imshow(high_noise, cmap='gray')
    plots[2, 0].set_title('Original Image- Highnoise Blurred')
    plots[2, 1].imshow(img_spec_3, cmap='gray')
    plots[2, 1].set_title('Image spectrum-High noise')
    plots[2, 2].imshow(kernel_spec_3, cmap='gray')
    plots[2, 2].set_title('kernel spectrum-High noise')
    plots[2, 3].imshow(final_spec_3, cmap='gray')
    plots[2, 3].set_title('Filtered spectrum-High noise')
    plots[2, 4].imshow(image_3, cmap='gray')
    plots[2, 4].set_title('Restored Image-High noise')
    plt.show()

    # part b
    wiener_low_noise = af.wiener_filter(low_noise, kernel, 1)
    wiener_med_noise = af.wiener_filter(med_noise, kernel, 5)
    wiener_high_noise = af.wiener_filter(high_noise, kernel, 10)
    fig, plots = plt.subplots(2, 3)
    fig.suptitle('Question 1 (c):Constrained Least Squares Filtering')
    plots[0, 0].imshow(low_noise, cmap='gray')
    plots[0, 0].set_title('Original Image- Low noise Blurred')
    plots[1, 0].imshow(wiener_low_noise, cmap='gray')
    plots[1, 0].set_title('CLSF filtered- Low noise')
    plots[0, 1].imshow(med_noise, cmap='gray')
    plots[0, 1].set_title('Original- Medium noise blurred')
    plots[1, 1].imshow(wiener_med_noise, cmap='gray')
    plots[1, 1].set_title('CLSF filtered- Med noise')
    plots[0, 2].imshow(high_noise, cmap='gray')
    plots[0, 2].set_title('Original- High noise blurred')
    plots[1, 2].imshow(wiener_high_noise, cmap='gray')
    plots[1, 2].set_title('CLSF filtered- High noise')
    plt.show()

    # # part c
    clsf_low_noise = af.clsf(low_noise, kernel, 0.1)
    clsf_med_noise = af.clsf(med_noise, kernel, 0.1)
    clsf_high_noise = af.clsf(high_noise, kernel, 0.1)
    fig, plots = plt.subplots(2, 3)
    fig.suptitle('Question 1 (c):Constrained Least Squares Filtering')
    plots[0, 0].imshow(low_noise, cmap='gray')
    plots[0, 0].set_title('Original Image- Low noise Blurred')
    plots[1, 0].imshow(clsf_low_noise, cmap='gray')
    plots[1, 0].set_title('CLSF filtered- Low noise')
    plots[0, 1].imshow(med_noise, cmap='gray')
    plots[0, 1].set_title('Original- Medium noise blurred')
    plots[1, 1].imshow(clsf_med_noise, cmap='gray')
    plots[1, 1].set_title('CLSF filtered- Med noise')
    plots[0, 2].imshow(high_noise, cmap='gray')
    plots[0, 2].set_title('Original- High noise blurred')
    plots[1, 2].imshow(clsf_high_noise, cmap='gray')
    plots[1, 2].set_title('CLSF filtered- High noise')
    plt.show()


def question_2():
    # part a
    noisy = io.imread('noisy-book1.png')
    gauss_denoised = af.gaussian_denoise(noisy, 7, 3)
    median_denoised = af.median_filter_denoise(noisy, 5)
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 2 (a):Comparing median and gaussian filtering')
    plots[0].imshow(noisy, cmap='gray')
    plots[0].set_title('Original Image')
    plots[1].imshow(gauss_denoised, cmap='gray')
    plots[1].set_title('Gaussian filtered')
    plots[2].imshow(median_denoised, cmap='gray')
    plots[2].set_title('Median filtered')
    plt.show()

    # part b
    noisy_2 = io.imread('noisy-book2.png')
    gauss_denoised = af.gaussian_denoise(noisy_2, 11, 6)
    bilateral_filtered = af.bilateral_filter(noisy_2, 11, 6, 0.3)
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
