import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np


def main():
    low_light_1 = io.imread('LowLight_1.png')
    low_light_2 = io.imread('LowLight_2.png')
    low_light_3 = io.imread('LowLight_3.png')
    stone_face = io.imread('StoneFace.png')
    hazy = io.imread('Hazy.png')
    books = io.imread('MathBooks.png')
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
    # saturated_stretch_image = saturated_contrast_stretch('MathBooks.png')
    # plot_side_by_side(books, saturated_stretch_image, 'saturated stretch')
    # resized_image = resize('LowLight_1.png', 2)
    # plot_side_by_side(low_light_1, resized_image, 'resizing')
    clahe_image = contrast_limited_histogram_equalize('StoneFace.png')
    plot_side_by_side(stone_face, clahe_image, 'tite')
    # bleh = np.array([0, 0, 1, 2, 5, 6, 12, 1, 2, 10])
    # contrast_limit_histogram(bleh)


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
    cdf = np.ones(256)
    cdf[:-1] = hist.cumsum() / no_total_pixels
    enhanced_image = cdf[image_data] * 255
    return enhanced_image


def contrast_limited_histogram_equalize(image_path):
    image_data = io.imread(image_path)
    enhanced_image = np.zeros_like(image_data)
    rows, columns = image_data.shape
    y_size = int(rows / 8)
    x_size = int(columns / 8)
    for i in range(0, rows, y_size):
        for j in range(0, columns, x_size):
            block = image_data[i:i + y_size, j:j + x_size]
            total_pixels = block.size
            hist, _ = np.histogram(block, bins=255, range=(0, 255))
            cdf = np.ones(256)
            cl_hist = contrast_limit_histogram(hist)
            cdf[:-1] = cl_hist.cumsum() / total_pixels
            enhanced_block = cdf[block] * 255
            enhanced_image[i:i + y_size, j:j + x_size] = enhanced_block
    return enhanced_image


def contrast_limit_histogram(hist_freq):
    threshold = 5
    # maximum iterations for which program can run to equalize histogram
    # to prevent infinite loops for very narrow histograms
    max_iterations = 100
    size = np.count_nonzero(hist_freq)
    for _ in range(max_iterations):
        if(hist_freq > threshold).sum() > 0:
            excess = (hist_freq[hist_freq > threshold] - threshold).sum()
            distribution_factor = excess // size
            hist_freq = hist_freq.clip(None, threshold)
            for i in range(len(hist_freq)):
                if hist_freq[i] > 0 and hist_freq[i] < threshold:
                    hist_freq[i] += distribution_factor
        else:
            break
    return hist_freq


def clahe_with_overlap(image_path):
    size = 8
    overlap = 0.25
    image_data = io.imread(image_path)
    enhanced_image = np.zeros_like(image_data)
    rows, columns = image_data.shape
    for i in range(int(size * (1 - overlap)), rows, size):
        for j in range(int(size * (1 - overlap)), columns, size):
            print(i, j)


def saturated_contrast_stretch(image_path):
    image_data = io.imread(image_path)
    percentage = 15
    enhanced_image = np.zeros_like(image_data)
    # for each of R,G,B channels
    for i in range(3):
        hist, _ = np.histogram(image_data[:, :, i], bins=255, range=(0, 255))
        total_pixels = image_data.size
        threshold_top, threshold_bottom = 1, 1
        channel = image_data[:, :, i]
        # find the value below which n% of pixels lie
        for value in range(255):
            percentage_bottom = ((hist[:value]).sum() / total_pixels) * 100
            if percentage_bottom > percentage:
                threshold_bottom = value
                break
        # find the value above which n% of pixels lie
        for value in range(255):
            percentage_top = (hist[255 - value:].sum() / total_pixels) * 100
            if percentage_top > percentage:
                threshold_top = value
                break
        # set n% brightest pixels to 255 and darkest to 0
        channel[channel < threshold_bottom] = 0
        channel[channel > threshold_top] = 255
        # perform linear contrast stretch on the channel
        gain = 255 / np.max(channel[channel < 255])
        enhanced_channel = channel * gain
        # values set to 255 would've gone above 255, bring them back to 255
        enhanced_channel[enhanced_channel > 255] = 255
        enhanced_image[:, :, i] = enhanced_channel
    return enhanced_image


def resize(image_path, resizing_factor, interpolation='nearest'):
    image_data = np.array([[1, 2], [3, 4]])
    # image_data = io.imread(image_path)
    rows, columns = image_data.shape
    rows_resized = int(rows * resizing_factor)
    columns_resized = int(columns * resizing_factor)
    resized_image = np.zeros((rows_resized, columns_resized))
    for i in range(rows_resized):
        for j in range(columns_resized):
            y = round(i / resizing_factor)
            x = round(j / resizing_factor)
            resized_image[i, j] = image_data[y, x]
    print(resized_image)
    return resized_image


def rotate(image_path, angle):
    pass


if __name__ == "__main__":
    main()
