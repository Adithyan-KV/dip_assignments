import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as col
from skimage.filters import gaussian
import scipy.ndimage as ndi


def main():
    window_image = io.imread('bar.jpg')
    window_img_grayscale = col.rgb2gray(window_image)
    corner_map = harris_corner_detect(window_img_grayscale)


def harris_corner_detect(image_data):
    image_blurred = gaussian(image_data, 2)
    Ix = ndi.sobel(image_blurred, 0)
    Iy = ndi.sobel(image_blurred, 1)

    IxIy = Ix * Iy
    Ix_sq = Ix**2
    Iy_sq = Iy**2

    rows, cols = image_data.shape
    R = np.zeros_like(image_data)
    k = 0.04

    win_size = 3
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

    print(R.max(), R.min())
    corner_map = np.abs(R) > 30

    fig, plots = plt.subplots(1, 3)
    fig.suptitle('Question 1 (c):Constrained Least Squares Filtering')
    plots[0].imshow(image_data, cmap='gray')
    plots[0].set_title('Original Image')
    plots[1].imshow(R, cmap='gray')
    plots[1].set_title('R')
    plots[2].imshow(corner_map, cmap='gray')
    plots[2].set_title('Corners detected')
    plt.show()

    return corner_map


if __name__ == "__main__":
    main()
