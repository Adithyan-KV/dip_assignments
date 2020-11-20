import skimage.io as io
import matplotlib.pyplot as plt
import math
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
    # resized_image_2 = resize('LowLight_1.png', 2, interpolation='bilinear')
    # plot_side_by_side(low_light_1, resized_image_2, 'resizing')
    # clahe_image = contrast_limited_histogram_equalize('StoneFace.png')
    # plot_side_by_side(stone_face, clahe_image, 'tite')
    # clahe_overlap_image = clahe_with_overlap('StoneFace.png', 0.25)
    # plot_side_by_side(stone_face, clahe_overlap_image, 'CLAHE overlap')
    rotated_image = rotate('StoneFace.png', 15)
    # rotated_image = ImgRotate('StoneFace.png', 90)
    plt.imshow(rotated_image, cmap='gray')
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
            block = image_data[i:i + y_size, j:j + x_size]
            total_pixels = block.size
            hist, _ = np.histogram(block, bins=256, range=(0, 255))
            cl_hist = contrast_limit_histogram(hist)
            cdf = cl_hist.cumsum() / total_pixels
            enhanced_block = cdf[block] * 255
            enhanced_image[i:i + y_size, j:j + x_size] = enhanced_block
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

            block = image_data[start_i:stop_i, start_j:stop_j]
            total_pixels = block.size
            hist, _ = np.histogram(block, bins=num_bins, range=(0, 255))
            cl_hist = contrast_limit_histogram(hist)
            cdf = cl_hist.cumsum() / total_pixels
            enhanced_block = np.zeros_like(block, dtype=np.float64)
            enhanced_block = cdf[block] * 255
            # add the present enhanced block to the total image
            enhanced_image[start_i:stop_i,
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
    percentage = 0.5
    fraction = int(percentage / 100 * total_pixels)

    # perform contrast stretch on all channels
    channels = image_data.shape[2]
    for i in range(channels):

        temp = image_data[:, :, i]

        hist, _ = np.histogram(temp, range=[0, 255], bins=256)

        min_thresh, max_thresh = get_threshold(hist, fraction)

        # set pixels above and below a threshold to white and black
        temp[temp >= max_thresh] = 255
        temp[temp <= min_thresh] = 0

        temp_mask_inv = np.logical_or((temp == 255), (temp == 0))
        temp_mask = np.logical_not(temp_mask_inv)
        masked_temp = temp_mask * temp
        masked_temp_inv = temp_mask_inv * temp

        min_intensity = np.min(masked_temp[np.nonzero(masked_temp)])
        max_intensity = np.max(masked_temp)

        enhanced_masked_temp = (
            masked_temp * (255 / (max_intensity - min_intensity)))
        enchanced_temp = enhanced_masked_temp + masked_temp_inv
        enhanced_image[:, :, i] = enchanced_temp

    return enhanced_image


def get_threshold(hist, fraction):

    int_sum = 0
    min_int = 0
    max_int = 255
    while int_sum < fraction:
        min_int += 1
        int_sum += hist[min_int]
    int_sum = 0
    while int_sum < fraction:
        max_int -= 1
        int_sum += hist[max_int]
    return [min_int, max_int]


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
                resized_image[i, j] = image_data[min(
                    y, rows - 1), min(x, columns - 1)]

            if interpolation == "bilinear":
                y = int(i / resize_factor)
                x = int(j / resize_factor)

                # if not edge pixel
                if 0 < y < rows - 1 and 0 < x < columns - 1:
                    # four neighboring image cordinates
                    f = np.array([image_data[y, x], image_data[y + 1, x],
                                  image_data[y, x + 1], image_data[y + 1, x + 1]])
                    n = np.array([[1, y, x, y * x], [1, y + 1, x, (y + 1) * x],
                                  [1, y, x + 1, y * (x + 1)], [1, y + 1, x + 1, (y + 1) * (x + 1)]])

                    # if the inverse exists
                    if np.linalg.det(n) != 0:
                        # solve and find bilinear weights
                        A = np.linalg.solve(n, f)
                        resized_image[i, j] = A[0] + A[1] * \
                            y + A[2] * x + A[3] * x * y

                    # if it cannot be solved for bilinear weights
                    else:
                        # set pixel as average of surrounding pixels
                        resized_image[i, j] = f.sum() / 4
                # if edge pixel assign the nearest neighbor
                else:
                    resized_image[i, j] = image_data[min(y, rows - 1),
                                                     min(x, columns - 1)]
    return resized_image


def rotate(image_path, angle, interpolation='nearest'):
    image_data = io.imread(image_path)
    h = image_data.shape[0]
    w = image_data.shape[1]
    # accounting for images with differing number of channels
    if len(image_data.shape) > 2:
        channels = image_data.shape[2]
    else:
        channels = 1
    theta = math.radians(angle)
    cosine = math.cos(theta)
    sine = math.sin(theta)

    # Dimensions of rotated_image
    new_h = round(abs(h * cosine) + abs(w * sine)) + 1
    new_w = round(abs(w * cosine) + abs(h * sine)) + 1
    rotated_image = np.zeros((new_h, new_w, channels), dtype=int)

    # center of image about which rotation will occur
    origin_h = round(((h + 1) / 2) - 1)
    origin_w = round(((w + 1) / 2) - 1)

    # center of new image
    new_origin_h = round(((new_h + 1) / 2) - 1)
    new_origin_w = round(((new_w + 1) / 2) - 1)

    for i in range(new_h):
        for j in range(new_w):
            # co-ordinates w.r.t center
            y = new_origin_h - i
            x = new_origin_w - j

            y_new = y * cosine - x * sine
            x_new = y * sine + x * cosine

            i_new = origin_h - y_new
            j_new = origin_w - x_new

            if 0 < i_new < h - 1 and 0 < j_new < w - 1:

                if interpolation == 'nearest':
                    rotated_image[i, j] = image_data[min(
                        round(i_new), h - 1), min(round(j_new), w - 1)]

                if interpolation == 'bilinear':
                    # if not edge pixel
                    if 0 < y < h - 1 and 0 < x < w - 1:
                        # four neighboring image cordinates
                        f = np.array([image_data[y, x], image_data[y + 1, x],
                                      image_data[y, x + 1], image_data[y + 1, x + 1]])
                        n = np.array([[1, y, x, y * x], [1, y + 1, x, (y + 1) * x],
                                      [1, y, x + 1, y * (x + 1)], [1, y + 1, x + 1, (y + 1) * (x + 1)]])

                        # if the inverse exists
                        if np.linalg.det(n) != 0:
                            # solve and find bilinear weights
                            A = np.linalg.solve(n, f)
                            rotated_image[i, j] = A[0] + A[1] * \
                                y + A[2] * x + A[3] * x * y

                        # if it cannot be solved for bilinear weights
                        else:
                            # set pixel as average of surrounding pixels
                            rotated_image[i, j] = f.sum() / 4
                    # if edge pixel assign the nearest neighbor
                    else:
                        rotated_image[i, j] = image_data[min(y, h - 1),
                                                         min(x, w - 1)]
    return rotated_image


def ImgRotate(image_path, angle, interpolation="nearest"):
    image = io.imread(image_path)  # Load the image

    image = image / 255  # Normalize teh image to [0,1]

    angle = math.radians(angle)  # Convert the angle to radians

    # Define the cos and sin
    cos = math.cos(angle)
    sin = math.sin(angle)

    height, width = image.shape  # Obtain the image dimensions

    # Compute the dimesions of the final image
    height_new = round(abs(height * cos + width * sin)) + 1
    width_new = round(abs(width * cos + height * sin)) + 1

    # Compute the center of the image about which we are to rotate
    height_center = round(height / 2)
    width_center = round(width / 2)

    # Commpute the correspoding center of the rotated image
    height_center_new = round(height_new / 2)
    width_center_new = round(width_new / 2)

    # Initialize an array to store the final image
    rotated_image = np.zeros((height_new, width_new))

    # Iterate through each pixel in the final image
    for i in range(height_new):
        for j in range(width_new):

            # Convert the final image indices to cartesion coordinates centered at image center
            y = height_center_new - i
            x = width_center_new - j

            # Obtain the corresponding inidces of the final image by performing roation
            y_ = y * cos - x * sin
            x_ = y * sin + x * cos

            i_ = height_center - y_
            j_ = x_ + width_center

            # print(i, j, x, y, x_, y_, i_, j_)

            if (i_ < 0) or (i_ > height - 1) or (j_ < 0) or (j_ > width - 1):
                continue

            # Case of nearest neighbour interpolation
            if interpolation == "nearest":

                # Round off the indices
                rotated_image[i, j] = image[min(
                    round(i_), height - 1), min(round(j_), width - 1)]

            # Case of bilinear interpolation
            elif interpolation == "bilinear":

                # Compute the four neighbours of (i_, j_)
                i_f = math.floor(i_)
                i_c = math.ceil(i_)
                j_f = math.floor(j_)
                j_c = math.ceil(j_)

                # Make sure the neighbour indices are in the range of the imae dimensions
                if (i_c in range(height)) and (j_c in range(width)):

                    # Create a matrix with the bilinear terms
                    a = np.array([[1, i_f, j_f, i_f * j_f], [1, i_f, j_c, i_f * j_c],
                                  [1, i_c, j_f, i_c * j_f], [1, i_c, j_c, i_c * j_c]])

                    # Create column vector with neighbour intensities
                    b = np.array([[image[i_f, j_f]], [image[i_f, j_c]], [
                                 image[i_c, j_f]], [image[i_c, j_c]]])

                    # Case where determinant of a is not zero, only when you'll get solution for the set of linear equations
                    if np.linalg.det(a) != 0:
                        # Obtain the bilinear coefficients by soolving ax=b
                        x = np.linalg.solve(a, b)

                        # Add the interpolated pixel value to the final image
                        rotated_image[i, j] = x[0] + x[1] * \
                            i_ + x[2] * j_ + x[3] * i_ * j_

                    # If the system of equations dont have a solution, set the corresponding pixel to the
                    # average of its neighbours
                    else:
                        rotated_image[i, j] = b.sum() / 4

    return rotated_image


if __name__ == "__main__":
    main()
