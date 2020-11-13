from skimage.util.shape import view_as_blocks
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np


def main():
    low_light_1 = io.imread('LowLight_1.png')
    low_light_2 = io.imread('LowLight_2.png')
    low_light_3 = io.imread('LowLight_3.png')
    stone_face = io.imread('StoneFace.png')
    hazy = io.imread('Hazy.png')
    linear_stretched_img_1 = linear_contrast_stretch('LowLight_1.png')
    linear_stretched_img_2 = linear_contrast_stretch('LowLight_2.png')
    # plot_side_by_side(low_light_1, linear_stretched_img_1, 'Linear stretch')
    # plot_side_by_side(low_light_2, linear_stretched_img_2, 'Linear stretch')

    power_stretched_image_1 = power_law_contrast_stretch(
        'LowLight_2.png', 1 / 2)
    power_stretched_image_2 = power_law_contrast_stretch('Hazy.png', 2)
    # plot_side_by_side(low_light_2, power_stretched_image_1,
    #   'Power law stretch')
    # plot_side_by_side(hazy, power_stretched_image_2,
    #   'Power law stretch')
    histogram_eq_image_1 = histogram_equalize('LowLight_2.png')
    # plot_side_by_side(low_light_2, histogram_eq_image_1, 'hist eq')
    histogram_eq_image_2 = histogram_equalize('LowLight_3.png')
    histogram_eq_image_3 = histogram_equalize('Hazy.png')
    histogram_eq_image_4 = histogram_equalize('StoneFace.png')
    # plot_side_by_side(low_light_3, histogram_eq_image_2, 'hist eq')
    # plot_side_by_side(hazy, histogram_eq_image_3, 'hist eq')
    # plot_side_by_side(stone_face, histogram_eq_image_4, 'hist eq')
    clahe_image_1 = contrast_limited_histogram_equalize('StoneFace.png')
    plt.imshow(clahe_image_1, cmap='gray')
    plt.show()


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


def histogram_equalize(image_path):
    image_data = io.imread(image_path)
    no_total_pixels = image_data.size
    hist, _ = np.histogram(image_data, bins=255, range=(0, 255))
    cdf = np.array([sum(hist[:i + 1]) / no_total_pixels for i in range(256)])
    enhanced_image = cdf[image_data] * 255
    return enhanced_image


def contrast_limited_histogram_equalize(image_path):
    size = 8
    image_data = io.imread(image_path)
    enhanced_image = np.zeros_like(image_data)
    rows, columns = image_data.shape
    for i in range(rows // size):
        for j in range(columns // size):
            block = image_data[i * size:(i + 1) *
                               size, j * size:(j + 1) * size]
            hist, _ = np.histogram(block, bins=255, range=(0, 255))
            cl_hist = contrast_limit_histogram(hist)
            cdf = np.array(
                [sum(cl_hist[:i + 1]) / size**2 for i in range(256)])
            enhanced_block = cdf[block] * 255
            enhanced_image[i * size:(i + 1) *
                           size, j * size:(j + 1) * size] = enhanced_block
    return enhanced_image


def contrast_limit_histogram(histogram_freq):
    threshold = 64
    hist_size = len(histogram_freq)
    excess = histogram_freq[histogram_freq > threshold].sum()
    distribute = excess // hist_size
    cl_histogram_freq = histogram_freq.clip(None, threshold) + distribute
    return cl_histogram_freq


if __name__ == "__main__":
    main()
