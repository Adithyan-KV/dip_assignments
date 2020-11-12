import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np


def main():
    original_img_1 = io.imread('LowLight_1.png')
    original_img_2 = io.imread('LowLight_2.png')
    original_img_3 = io.imread('Hazy.png')
    linear_stretched_img_1 = linear_contrast_stretch('LowLight_1.png')
    linear_stretched_img_2 = linear_contrast_stretch('LowLight_2.png')
    plot_side_by_side(original_img_1, linear_stretched_img_1, 'Linear stretch')
    plot_side_by_side(original_img_2, linear_stretched_img_2, 'Linear stretch')

    power_stretched_image_1 = power_law_contrast_stretch(
        'LowLight_2.png', 1 / 2)
    power_stretched_image_2 = power_law_contrast_stretch('Hazy.png', 2)
    plot_side_by_side(original_img_2, power_stretched_image_1,
                      'Power law stretch')
    plot_side_by_side(original_img_3, power_stretched_image_2,
                      'Power law stretch')


def plot_side_by_side(image_1, image_2, title):
    fig, plots = plt.subplots(1, 2)
    fig.suptitle(title)
    plots[0].imshow(image_1, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(image_2, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title('Contrast Enhanced')
    plt.show()


def linear_contrast_stretch(image_path):
    image_data = io.imread(image_path)
    A = np.max(image_data)
    k = 255
    gain = k / A
    stretched_image = image_data * gain
    return stretched_image


def power_law_contrast_stretch(image_path, power):
    image_data = io.imread(image_path)
    normalized_image_data = image_data / 255
    enhanced_image = normalized_image_data**(power) * 255
    return enhanced_image


if __name__ == "__main__":
    main()
