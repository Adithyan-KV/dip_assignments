import skimage.io as io
import skimage.filters as flt
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


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


def square_average_filter(image_path, size):
    image_data = io.imread(image_path)
    kernel = np.ones((size, size)) / (size**2)
    denoised_image = ndi.convolve(image_data, kernel, mode='reflect')
    return denoised_image


def high_boost_filter(image):
    original_image = io.imread('./characters.tif')
    blurred_image = flt.gaussian(image, 5) * 255
    mask = image - blurred_image
    # mask = mask - np.min(mask)
    plt.imshow(mask, cmap='gray')
    plt.figure()
    k_values = np.arange(-10, 10, 0.1)
    errors = np.zeros_like(k_values)
    for i in range(len(k_values)):
        k = k_values[i]
        sharpened_image = image + k * mask
        error = np.square(sharpened_image - original_image).mean()
        errors[i] = error
    k_optimum = k_values[np.argmin(errors)]
    sharpened_image = image + k_optimum * mask
    print(k_optimum)
    plt.plot(k_values, errors)
    plt.show()
    return sharpened_image


def generate_sinusoidal_image(M, N):
    pass


if __name__ == "__main__":
    main()
