import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as col
from skimage.filters import gaussian
import skimage.transform as tr
import scipy.ndimage as ndi
from skimage.util import random_noise


def main():
    window_image = io.imread('win.jpg')
    car_image = io.imread('car.jpg')
    car_img_grayscale = col.rgb2gray(car_image)
    window_img_grayscale = col.rgb2gray(window_image)
    image_rot_1, locations_1 = rotate_and_corner_detect(car_img_grayscale, 45)
    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 1 :Harris corner detector')
    plots[0, 0].imshow(image_rot_1, cmap='gray')
    plots[0, 0].set_title('Original Image')
    plots[0, 1].imshow(image_rot_1, cmap='gray')
    plots[0, 1].set_title('R')
    plots[0, 1].scatter(locations_1[1], locations_1[0], s=5, c='yellow')
    plt.show()


def harris_corner_detect(image_data, win_size=3, blur_std=2, threshold=0.2):
    image_blurred = gaussian(image_data, 2)
    Ix = ndi.sobel(image_blurred, 0)
    Iy = ndi.sobel(image_blurred, 1)

    IxIy = Ix * Iy
    Ix_sq = Ix**2
    Iy_sq = Iy**2

    rows, cols = image_data.shape
    R = np.zeros_like(image_data)
    k = 0.04

    offset = int(win_size / 2)

    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):

            start_x, end_x = i - offset, i + offset + 1
            start_y, end_y = j - offset, j + offset + 1

            Ix_sq_win = Ix_sq[start_x:end_x, start_y:end_y]
            Iy_sq_win = Iy_sq[start_x:end_x, start_y:end_y]
            IxIy_win = IxIy[start_x:end_x, start_y:end_y]

            Sx_sq = Ix_sq_win.sum()
            Sy_sq = Iy_sq_win.sum()
            Sxy = IxIy_win.sum()

            M = np.array([[Sx_sq, Sxy],
                          [Sxy, Sy_sq]])

            R[i, j] = np.linalg.det(M) - k * (np.trace(M)**2)

    corner_map = R > threshold * R.max()
    corner_locations = np.where(corner_map == 1)

    return R, corner_map, corner_locations


def visualize_multiple_thresholds(image_data, R_map):
    thresholds = np.arange(0.05, 1, 0.15)
    fig, plots = plt.subplots(2, 3)
    fig.suptitle('Question 1 : Multiple Thresholds')
    for i in range(0, len(thresholds) - 1):
        corner_map = R_map > thresholds[i]
        locations = np.where(corner_map == 1)

        # coordinates for subplots in plotting
        row = int(i / 3)
        column = i % 3

        plots[row, column].imshow(image_data, cmap='gray')
        plots[row, column].scatter(
            locations[1], locations[0], s=5, c='red')
        plots[row, column].set_title(
            f'Threshold {round(thresholds[i]*100)}% of max value')
    plt.show()


def rotate_and_corner_detect(image, angle):
    image_rot = tr.rotate(image, angle, resize=True)
    _, _, corner_locations = harris_corner_detect(image_rot, 3, 2, 0.1)
    return image_rot, corner_locations


def scale_and_corner_detect(image, scale_factor):
    image_scaled = tr.rescale(image, scale_factor, anti_aliasing=True)
    _, _, corner_locations = harris_corner_detect(image_scaled, 3, 2, 0.1)
    return image_scaled, corner_locations


def add_noise_and_corner_detect(image, noise='gaussian', std=0.01, am=0.1):
    if noise == 'gaussian':
        image_noisy = random_noise(image, noise, var=std)
    elif noise == 's&p':
        image_noisy = random_noise(image, noise, amount=am)
    _, _, corner_locations = harris_corner_detect(image_noisy, 3, 2, 0.1)
    return image_noisy, corner_locations


if __name__ == "__main__":
    main()
