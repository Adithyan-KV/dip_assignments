import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np


def main():
    linear_contrast_strech('LowLight_1.png')


def linear_contrast_strech(image_path):
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
    plt.imshow(stretched_image, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
