import AllFunctions as af
import matplotlib.pyplot as plt
import skimage.io as io


def main():
    # Question 1
    pass

    # Question 2
    # part a
    noisy = io.imread('noisy-book1.png')
    gauss_denoised = af.gaussian_denoise(noisy, 7, 3)
    median_denoised = af.median_filter_denoise(noisy, 5)
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Comparing median and gaussian filtering')
    plots[0].imshow(noisy, cmap='gray')
    plots[0].set_title('Original Image')
    plots[1].imshow(gauss_denoised, cmap='gray')
    plots[1].set_title('Gaussian filtered')
    plots[2].imshow(median_denoised, cmap='gray')
    plots[2].set_title('Median filtered')
    plt.show()


if __name__ == "__main__":
    main()
