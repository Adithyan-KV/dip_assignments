import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np


def main():
    original_img_1 = io.imread('LowLight_1.png')
    original_img_2 = io.imread('LowLight_2.png')
    linear_stretched_img_1 = linear_contrast_stretch('LowLight_1.png')
    linear_stretched_img_2 = linear_contrast_stretch('LowLight_2.png')
    plot_side_by_side(original_img_1, linear_stretched_img_1)
    plot_side_by_side(original_img_2, linear_stretched_img_2)


def plot_side_by_side(image_1, image_2):
    _, plots = plt.subplots(1, 2)
    plots[0].imshow(image_1, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(image_2, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title('Contrast Enhanced')
    plt.show()


def linear_contrast_stretch(image_path):
    image_data = io.imread(image_path)
    A = np.max(image_data)
    B = np.min(image_data)
    k = 255
    if (A - B) != 0:
        gain = k / (A - B)
    # if image is one uniform intensity, no contrast enhancement possible
    else:
        gain = 1
    stretched_image = image_data * gain
    return stretched_image


if __name__ == "__main__":
    main()
