import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


def main():
    noisy = io.imread('./noisy.tif')

    # Question 1 (a)
    denoised_1 = square_average_filter('./noisy.tif',5)
    denoised_2 = square_average_filter('./noisy.tif',10)
    denoised_3 = square_average_filter('./noisy.tif',15)
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


def square_average_filter(image_path, size):
    image_data = io.imread(image_path)
    kernel = np.ones((size,size))/(size**2)
    denoised_image = ndi.convolve(image_data, kernel, mode='reflect')
    return denoised_image


if __name__ == "__main__":
    main()
