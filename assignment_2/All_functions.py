import skimage.io as io
import matplotlib.pyplot as plt
import math
import numpy as np


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
    hist, _ = np.histogram(image_data, bins=256, range=(0, 255))
    cdf = hist.cumsum() / no_total_pixels
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
            block = image_data[i:min(i + y_size, rows),
                               j: min(j + x_size, columns)]
            total_pixels = block.size
            hist, _ = np.histogram(block, bins=256, range=(0, 255))
            cl_hist = contrast_limit_histogram(hist)
            cdf = cl_hist.cumsum() / total_pixels
            enhanced_block = cdf[block] * 255
            enhanced_image[i: i + y_size, j: j + x_size] = enhanced_block
    return enhanced_image


def contrast_limit_histogram(hist_freq):
    threshold = 0.1 * hist_freq.sum()
    # maximum iterations for which program can run to equalize histogram
    # to prevent infinite loops for very narrow histograms
    max_iterations = 100
    size = np.count_nonzero(hist_freq)
    for _ in range(max_iterations):
        if(hist_freq > threshold).sum() > 0:
            # distribute the pixels over threshold equally over other values
            excess = (hist_freq[hist_freq > threshold] - threshold).sum()
            distribution_factor = excess // size
            hist_freq = hist_freq.clip(None, threshold)
            for i in range(len(hist_freq)):
                if hist_freq[i] > 0 and hist_freq[i] < threshold:
                    hist_freq[i] += distribution_factor
        else:
            break
    return hist_freq


def clahe_with_overlap(image_path, overlap=0.25):
    image_data = io.imread(image_path)
    num_bins = 256
    rows, columns = image_data.shape
    tiles = 8

    # calculating some values to split the image into blocks
    block_size_x = math.ceil(columns / (tiles - (tiles * overlap) + overlap))
    block_size_y = math.ceil(rows / (tiles - (tiles * overlap) + overlap))
    overlap_x = math.ceil(block_size_x * overlap)
    overlap_y = math.ceil(block_size_y * overlap)
    non_overlap_x = block_size_x - overlap_x
    non_overlap_y = block_size_y - overlap_y

    # An array to keep track of regions with overlap
    overlap_map = np.zeros_like(image_data)

    # The final contrast enhanced image
    enhanced_image = np.zeros_like(image_data, dtype=np.float64)

    # Perform histogram equalization on all tiles
    for i in range(tiles):
        for j in range(tiles):

            start_i = i * non_overlap_y
            stop_i = min(start_i + block_size_y, rows)
            start_j = j * non_overlap_x
            stop_j = min(start_j + block_size_x, columns)

            block = image_data[start_i: stop_i, start_j: stop_j]
            total_pixels = block.size
            hist, _ = np.histogram(block, bins=num_bins, range=(0, 255))
            cl_hist = contrast_limit_histogram(hist)
            cdf = cl_hist.cumsum() / total_pixels
            enhanced_block = np.zeros_like(block, dtype=np.float64)
            enhanced_block = cdf[block] * 255
            # add the present enhanced block to the total image
            enhanced_image[start_i: stop_i,
                           start_j: stop_j] += enhanced_block
            # keep track of the overlapping areas
            overlap_map[start_i: stop_i, start_j: stop_j] += 1

    # take average in areas with overlap
    enhanced_image = np.divide(enhanced_image, overlap_map)

    return enhanced_image


def saturated_contrast_stretch(image_path):
    image_data = io.imread(image_path)
    enhanced_image = np.zeros_like(image_data)
    total_pixels = image_data.size
    percentage = 1
    fraction = int(percentage / 100 * total_pixels)

    # perform contrast stretch on all channels
    num_channels = image_data.shape[2]
    for i in range(num_channels):

        channel = image_data[:, :, i]

        hist, _ = np.histogram(channel, range=[0, 255], bins=256)

        # set pixels above and below a threshold to white and black
        min_thresh, max_thresh = get_threshold(hist, fraction)
        channel[channel >= max_thresh] = 255
        channel[channel <= min_thresh] = 0

        # masking out all the pixels that have 0 or 255 value
        mask_inverse = np.logical_or((channel == 255), (channel == 0))
        mask = np.logical_not(mask_inverse)
        masked_channel = mask * channel
        inverse_masked_channel = mask_inverse * channel

        # performing contrast stretch on the rest
        min_intensity = np.min(masked_channel[np.nonzero(masked_channel)])
        max_intensity = np.max(masked_channel)
        gain = 255 / (max_intensity - min_intensity)
        enhanced_masked_channel = (masked_channel * gain)

        # add back masked 0 and 255 values
        enhanced_channel = enhanced_masked_channel + inverse_masked_channel
        enhanced_channel = enhanced_channel.clip(None, 255)
        enhanced_image[:, :, i] = enhanced_channel

    return enhanced_image


def get_threshold(hist, fraction):

    int_sum = 0
    minimum = 0
    while int_sum < fraction:
        minimum += 1
        int_sum += hist[minimum]
    int_sum = 0
    maximum = 255
    while int_sum < fraction:
        maximum -= 1
        int_sum += hist[maximum]
    return (minimum, maximum)


def resize(image_path, resize_factor, interpolation="nearest"):
    image_data = io.imread(image_path)

    rows, columns = image_data.shape

    # calculating final size of resized image
    columns_resized = int(columns * resize_factor)
    rows_resized = int(rows * resize_factor)
    resized_image = np.zeros((rows_resized, columns_resized))

    for i in range(rows_resized):
        for j in range(columns_resized):

            if interpolation == "nearest":
                y = round(i / resize_factor)
                x = round(j / resize_factor)
                # accounting for pixels in the last row
                resized_image[i, j] = image_data[min(y, rows - 1),
                                                 min(x, columns - 1)]

            if interpolation == "bilinear":
                y_ceil = math.ceil(i / resize_factor)
                x_ceil = math.ceil(j / resize_factor)
                y_floor = math.floor(i / resize_factor)
                x_floor = math.floor(j / resize_factor)

                # if not edge pixel
                if 0 < y_floor < rows - 1 and 0 < x_floor < columns - 1:
                    # four neighboring image cordinates
                    f = np.array([image_data[y_floor, x_floor],
                                  image_data[y_ceil, x_floor],
                                  image_data[y_floor, x_ceil],
                                  image_data[y_ceil, x_ceil]],)
                    n = np.array([[1, y_floor, x_floor, y_floor * x_floor],
                                  [1, y_ceil, x_floor, y_ceil * x_floor],
                                  [1, y_floor, x_ceil, y_floor * x_ceil],
                                  [1, y_ceil, x_ceil, y_ceil * x_ceil]])

                    # if the inverse exists
                    if np.linalg.det(n) != 0:
                        # solve and find bilinear weights
                        A = np.linalg.solve(n, f)
                        resized_image[i, j] = A[0] + A[1] * \
                            y_floor + A[2] * x_floor + A[3] * x_floor * y_floor

                    # if it cannot be solved for bilinear weights
                    else:
                        # set pixel as average of surrounding pixels
                        resized_image[i, j] = f.sum() / 4
                # if edge pixel assign the nearest neighbor
                else:
                    resized_image[i, j] = image_data[min(y_floor, rows - 1),
                                                     min(x_floor, columns - 1)]
    return resized_image


def ImgRotate(image_path, angle, interpolation="nearest"):
    image_data = io.imread(image_path)
    image_data = image_data

    # calculating some constants
    theta = math.radians(angle)
    cos = math.cos(theta)
    sin = math.sin(theta)

    h, w = image_data.shape

    # calculating the size of the rotated image
    height_new = round(abs(h * cos + w * sin)) + 1
    width_new = round(abs(w * cos + h * sin)) + 1

    # Center of original and rotated image coordinate systems
    origin_h = int((h + 1) / 2)
    origin_w = int((w + 1) / 2)
    origin_h_rot = int((height_new + 1) / 2)
    origin_w_rot = int((width_new + 1) / 2)

    # final rotated image
    rotated_image = np.zeros((height_new, width_new))

    for i in range(height_new):
        for j in range(width_new):

            # transform to image center coordinates
            y = origin_h_rot - i
            x = j - origin_w_rot

            # corresponding coords in rotated coordinates
            y_rot = y * cos - x * sin
            x_rot = y * sin + x * cos

            i_rot = origin_h - y_rot
            j_rot = x_rot + origin_w

            if 0 < i_rot < h - 1 and 0 < j_rot < w - 1:
                if interpolation == "nearest":
                    rotated_image[i, j] = image_data[min(
                        round(i_rot), h - 1), min(round(j_rot), w - 1)]

                elif interpolation == "bilinear":

                    # used to calculate neighboring pixels
                    i_floor = math.floor(i_rot)
                    j_floor = math.floor(j_rot)
                    i_ceil = math.ceil(i_rot)
                    j_ceil = math.ceil(j_rot)

                    if 0 < i_floor < h - 1 and 0 < j_floor < w - 1:

                        # used to solve for bilinear weights
                        n = np.array([[1, i_floor, j_floor, i_floor * j_floor],
                                      [1, i_floor, j_ceil, i_floor * j_ceil],
                                      [1, i_ceil, j_floor, i_ceil * j_floor],
                                      [1, i_ceil, j_ceil, i_ceil * j_ceil]])
                        f = np.array([[image_data[i_floor, j_floor]],
                                      [image_data[i_floor, j_ceil]],
                                      [image_data[i_ceil, j_floor]],
                                      [image_data[i_ceil, j_ceil]]])

                        # if solution exists
                        if np.linalg.det(n) != 0:
                            # solve for weights and update values
                            A = np.linalg.solve(n, f)
                            rotated_image[i, j] = A[0] + A[1] * \
                                i_rot + A[2] * j_rot + A[3] * i_rot * j_rot

                        # if cannot be solved
                        else:
                            # assign average value of surrounding pixels
                            rotated_image[i, j] = f.sum() / 4
    return rotated_image


def main():

    # loading in all the images
    low_light_1 = io.imread('LowLight_1.png')
    low_light_2 = io.imread('LowLight_2.png')
    low_light_3 = io.imread('LowLight_3.png')
    stone_face = io.imread('StoneFace.png')
    hazy = io.imread('Hazy.png')
    books = io.imread('MathBooks.png')

    # question 1 (a)
    linear_stretched_img_1 = linear_contrast_stretch('LowLight_1.png')
    linear_stretched_img_2 = linear_contrast_stretch('LowLight_2.png')
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 1:Linear contrast stretch')
    plots[0, 0].imshow(low_light_1, cmap='gray', vmax=255, vmin=0)
    plots[0, 0].set_title('Original image')
    plots[0, 1].imshow(linear_stretched_img_1, cmap='gray', vmax=255, vmin=0)
    plots[0, 1].set_title('Contrast enhanced')
    plots[1, 0].imshow(low_light_2, cmap='gray', vmin=0, vmax=255)
    plots[1, 0].set_title('Original image')
    plots[1, 1].imshow(linear_stretched_img_2, cmap='gray', vmax=255, vmin=0)
    plots[1, 1].set_title('Contrast enhanced')
    plt.show()

    # question 1 (b)
    power_stretched_image_1 = power_law_contrast_stretch(
        'LowLight_1.png', 1 / 2)
    power_stretched_image_2 = power_law_contrast_stretch(
        'LowLight_2.png', 1 / 2)
    power_stretched_image_3 = power_law_contrast_stretch('Hazy.png', 3)
    fig, plots = plt.subplots(3, 2)
    fig.suptitle('Question 1:Power Law contrast stretch')
    plots[0, 0].imshow(low_light_1, cmap='gray', vmax=255, vmin=0)
    plots[0, 0].set_title('Original image')
    plots[0, 1].imshow(power_stretched_image_1, cmap='gray', vmax=255, vmin=0)
    plots[0, 1].set_title('Contrast enhanced')
    plots[1, 0].imshow(low_light_2, cmap='gray', vmin=0, vmax=255)
    plots[1, 0].set_title('Original image')
    plots[1, 1].imshow(power_stretched_image_2, cmap='gray', vmax=255, vmin=0)
    plots[1, 1].set_title('Contrast enhanced')
    plots[2, 0].imshow(hazy, cmap='gray', vmin=0, vmax=255)
    plots[2, 0].set_title('Original image')
    plots[2, 1].imshow(power_stretched_image_3, cmap='gray', vmax=255, vmin=0)
    plots[2, 1].set_title('Contrast enhanced')
    plt.show()

    # question 1 (c)

    histogram_eq_image_1 = histogram_equalize('LowLight_2.png')
    histogram_eq_image_2 = histogram_equalize('Hazy.png')
    histogram_eq_image_3 = histogram_equalize('StoneFace.png')
    histogram_eq_image_4 = histogram_equalize('LowLight_3.png')
    fig, plots = plt.subplots(4, 2)
    fig.suptitle('Question 1:Histogram equalization')
    plots[0, 0].imshow(low_light_2, cmap='gray', vmax=255, vmin=0)
    plots[0, 0].set_title('Original image')
    plots[0, 1].imshow(histogram_eq_image_1, cmap='gray', vmax=255, vmin=0)
    plots[0, 1].set_title('Contrast enhanced')
    plots[1, 0].imshow(hazy, cmap='gray', vmin=0, vmax=255)
    plots[1, 0].set_title('Original image')
    plots[1, 1].imshow(histogram_eq_image_2, cmap='gray', vmax=255, vmin=0)
    plots[1, 1].set_title('Contrast enhanced')
    plots[2, 0].imshow(stone_face, cmap='gray', vmin=0, vmax=255)
    plots[2, 0].set_title('Original image')
    plots[2, 1].imshow(histogram_eq_image_3, cmap='gray', vmax=255, vmin=0)
    plots[2, 1].set_title('Contrast enhanced')
    plots[3, 0].imshow(low_light_3, cmap='gray', vmin=0, vmax=255)
    plots[3, 0].set_title('Original image')
    plots[3, 1].imshow(histogram_eq_image_4, cmap='gray', vmax=255, vmin=0)
    plots[3, 1].set_title('Contrast enhanced')
    plt.show()

    # question 1 (d)
    clahe_image = contrast_limited_histogram_equalize('StoneFace.png')
    clahe_overlap_image = clahe_with_overlap('StoneFace.png', 0.25)
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 1 (d):CLAHE')
    plots[0].imshow(stone_face, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(clahe_image, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title('CLAHE-without overlap')
    plots[2].imshow(clahe_overlap_image, cmap='gray', vmin=0, vmax=255)
    plots[2].set_title('CLAHE- with overlap(25%)')
    plt.show()

    # question 2
    saturated_stretch_image = saturated_contrast_stretch('MathBooks.png')
    fig, plots = plt.subplots(2)
    fig.suptitle('Question 2: Saturated contrast stretch')
    plots[0].imshow(books, vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(saturated_stretch_image, vmax=255, vmin=0)
    plots[1].set_title('Contrast stretched image')
    plt.show()

    # question 3
    resizing_factor = 2
    resized_image = resize('StoneFace.png', resizing_factor)
    resized_image_2 = resize(
        'StoneFace.png', resizing_factor, interpolation='bilinear')
    fig, plots = plt.subplots(1, 3, gridspec_kw={'width_ratios': [
        1, resizing_factor, resizing_factor], })
    fig.suptitle('Question 3:Resize image')
    plots[0].imshow(stone_face, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(resized_image, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title(
        f'resized {resizing_factor}x:Nearest neighbor')
    plots[2].imshow(resized_image_2, cmap='gray', vmin=0, vmax=255)
    plots[2].set_title(f'resized {resizing_factor}x:Bilinear')
    plt.show()

    # question 4
    angle = 45
    rotated_image = ImgRotate('StoneFace.png', angle, interpolation='nearest')
    rotated_image_2 = ImgRotate(
        'StoneFace.png', angle, interpolation='bilinear')
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 4:Rotate image')
    plots[0].imshow(stone_face, cmap='gray', vmax=255, vmin=0)
    plots[0].set_title('Original image')
    plots[1].imshow(rotated_image, cmap='gray', vmax=255, vmin=0)
    plots[1].set_title(f'Rotated by {angle} degrees: Nearest neighbor')
    plots[2].imshow(rotated_image_2, cmap='gray', vmin=0, vmax=255)
    plots[2].set_title(f'Rotated by {angle} degrees: Bilinear')
    plt.show()


if __name__ == "__main__":
    main()
