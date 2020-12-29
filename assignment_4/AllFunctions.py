import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import skimage.io as io
from scipy.io import loadmat


def main():
    kernel_obj = loadmat('BlurKernel.mat')
    kernel = kernel_obj['h']
    low_noise = io.imread('Blurred-LowNoise.png')
    filtered = inverse_filter(low_noise, kernel)


def inverse_filter(image_data, kernel):
    pass


def dft(image_data):
    dft_image = fft.fft2(image_data)
    dft_centered = fft.fftshift(dft_image)
    dft_spectrum = np.log(1 + np.abs(dft_centered))
    return dft_centered, dft_spectrum


def inverse_dft(dft_data):
    dft_decentralized = fft.ifftshift(dft_data)
    idft = fft.ifft2(dft_decentralized)
    filtered_image = np.abs(idft)
    return filtered_image


if __name__ == "__main__":
    main()
