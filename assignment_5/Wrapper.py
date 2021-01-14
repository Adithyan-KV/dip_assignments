import skimage.io as io
import matplotlib.pyplot as plt
import skimage.color as col
import AllFunctions as af


def main():
    question_1()
    # question_2()


def question_1():
    window_image = io.imread('win.jpg')
    car_image = io.imread('car.jpg')
    car_img_grayscale = col.rgb2gray(car_image)
    window_img_grayscale = col.rgb2gray(window_image)
    R_values_1, corner_map_1, locations_1 = af.harris_corner_detect(
        window_img_grayscale, 3, 2, 0.1)
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 1 :Harris corner detector')
    plots[0, 0].imshow(window_img_grayscale, cmap='gray')
    plots[0, 0].set_title('Original Image')
    plots[0, 1].imshow(R_values_1, cmap='gray')
    plots[0, 1].set_title('R')
    plots[1, 0].imshow(corner_map_1, cmap='gray')
    plots[1, 0].set_title('Thresholded map')
    plots[1, 1].imshow(window_image, cmap='gray')
    plots[1, 1].scatter(locations_1[1], locations_1[0], s=5, c='yellow')
    plots[1, 1].set_title('Corners detected')
    plt.show()

    R_values_2, corner_map_2, locations_2 = af.harris_corner_detect(
        car_img_grayscale, 3, 2, 0.1)
    fig, plots = plt.subplots(2, 2)
    fig.suptitle('Question 1 :Harris corner detector')
    plots[0, 0].imshow(car_img_grayscale, cmap='gray')
    plots[0, 0].set_title('Original Image')
    plots[0, 1].imshow(R_values_2, cmap='gray')
    plots[0, 1].set_title('R')
    plots[1, 0].imshow(corner_map_2, cmap='gray')
    plots[1, 0].set_title('Thresholded map')
    plots[1, 1].imshow(car_image, cmap='gray')
    plots[1, 1].scatter(locations_2[1], locations_2[0], s=5, c='red')
    plots[1, 1].set_title('Corners detected')
    plt.show()

    # visualizing multiple thresholds
    af.visualize_multiple_thresholds(car_image, R_values_2)


if __name__ == "__main__":
    main()
