import numpy as np
import scipy.ndimage as ndi
import scipy.fft as fft
import skimage.io as io
import skimage.filters as flt
import math
import matplotlib.pyplot as plt


def main():
    noisy = io.imread('./noisy.tif')

    # Question 1 (a)
    denoised_1 = square_average_filter('./noisy.tif', 5)
    denoised_2 = square_average_filter('./noisy.tif', 10)
    denoised_3 = square_average_filter('./noisy.tif', 15)
    # fig, plots = plt.subplots(1, 4)
    # fig.suptitle('Question 1 (a):Average filter')
    # plots[0].imshow(noisy, cmap='gray', vmax=255, vmin=0)
    # plots[0].set_title('Original image')
    # plots[1].imshow(denoised_1, cmap='gray', vmax=255, vmin=0)
    # plots[1].set_title('Size = 5')
    # plots[2].imshow(denoised_2, cmap='gray', vmax=255, vmin=0)
    # plots[2].set_title('Size = 10')
    # plots[3].imshow(denoised_3, cmap='gray', vmin=0, vmax=255)
    # plots[3].set_title('Size = 15')
    # plt.show()

    # sharpened = high_boost_filter(denoised_2)
    # fig, plots = plt.subplots(1, 3)
    # fig.suptitle('Question 1 (b):Sharpening')
    # plots[0].imshow(noisy, cmap='gray', vmax=255, vmin=0)
    # plots[0].set_title('Original image')
    # plots[1].imshow(denoised_2, cmap='gray', vmax=255, vmin=0)
    # plots[1].set_title('Denoised image')
    # plots[2].imshow(sharpened, cmap='gray', vmax=255, vmin=0)
    # plots[2].set_title('Sharpened image')
    # plt.show()

    # # Question 2(a)
    # sinusoidal_image = generate_sinusoidal_image(1001, 1001)
    # dft_image = dft(sinusoidal_image)
    # fig, plots = plt.subplots(1, 2)
    # fig.suptitle('Question 2 (a):DFT')
    # plots[0].imshow(sinusoidal_image, cmap='gray', vmax=255, vmin=0)
    # plots[0].set_title('Sinusoidal image')
    # plots[1].imshow(dft_image, cmap='gray', vmax=255, vmin=0)
    # plots[1].set_title('DFT spectrum')
    # plt.show()

    # Question 2(b)
    char_image = io.imread('./characters.tif')
    dft_data, dft_spectrum = dft(char_image)
    filtered_dft, filtered_spectrum = filter_low_pass(dft_data, 150)
    filtered_image = inverse_dft(filtered_dft)
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 2 (b):Low pass filtering')
    plots[0].imshow(dft_spectrum, cmap='gray')
    plots[0].set_title('Centered DFT spectrum')
    plots[1].imshow(filtered_spectrum, cmap='gray')
    plots[1].set_title('Low pass filtered Spectrum')
    plots[2].imshow(filtered_image, cmap='gray')
    plots[2].set_title('Filtered Image')
    plt.show()


def square_average_filter(image_path, size):
    image_data = io.imread(image_path)
    kernel = np.ones((size, size)) / (size**2)
    denoised_image = ndi.convolve(image_data, kernel, mode='reflect')
    return denoised_image


def high_boost_filter(image):
    original_image = io.imread('./characters.tif')
    blurred_image = flt.gaussian(image, 5) * 255
    mask = image - blurred_image
    # plt.imshow(mask, cmap='gray')
    # plt.figure()
    k_values = np.arange(-4, 4, 0.1)
    errors = np.zeros_like(k_values)
    for i in range(len(k_values)):
        k = k_values[i]
        sharpened_image = image + k * mask
        error = np.square(sharpened_image - original_image).mean()
        errors[i] = error
    k_optimum = k_values[np.argmin(errors)]
    print(np.argmin(errors))
    sharpened_image = image + k_optimum * mask
    print(k_optimum)
    plt.plot(k_values, errors)
    plt.show()
    return sharpened_image


def generate_sinusoidal_image(M, N):
    sin_image = np.zeros((M, N))
    rows, columns = sin_image.shape
    u = 100
    v = 200
    for m in range(rows):
        for n in range(columns):
            sin_image[m, n] = math.sin(2 * math.pi * (u * m / M + v * n / N))
    return sin_image * 255


def dft(image):
    dft = fft.fft2(image)
    dft_centered = fft.fftshift(dft)
    dft_spectrum = np.log(1 + np.abs(dft_centered)) * 255
    # dft_spectrum = np.log(1 + np.abs(dft)) * 255
    return dft_centered, dft_spectrum


def inverse_dft(dft_data):
    fft_decentralized = fft.ifftshift(dft_data)
    # plt.imshow(np.log(1 + np.abs(fft_decentralized)), cmap='gray')
    # plt.show()
    idft = fft.ifft2(fft_decentralized)
    filtered_image = np.abs(idft) * 255
    return filtered_image


def filter_low_pass(dft_centered, D_0=100):
    H = np.zeros_like(dft_centered)
    P, Q = dft_centered.shape
    dft_spectrum = np.log(1 + np.abs(dft_centered)) * 255
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            if D > D_0:
                H[u, v] = 0
            else:
                H[u, v] = 1
    filtered_dft = H * dft_centered
    filtered_spectrum = np.abs(H) * dft_spectrum
    return filtered_dft, filtered_spectrum


if __name__ == "__main__":
    main()
