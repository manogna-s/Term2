import numpy as np
import matplotlib.pyplot as plt


def get_padded_image(bin_image: np.ndarray, kernel: tuple) -> np.ndarray:
    h, w = bin_image.shape
    pad_h, pad_w = kernel[0] // 2, kernel[1] // 2
    padded_im = np.ones((h + pad_h * 2, w + pad_w * 2))
    padded_im[pad_h:pad_h + h, pad_w:pad_w + w] = bin_image
    # pad rows at the top and bottom
    padded_im[:pad_h] = padded_im[pad_h]
    padded_im[-pad_h:] = padded_im[-pad_h - 1]
    # pad columns at the left and right end
    padded_im[:, :pad_w] = np.expand_dims(padded_im[:, pad_w], 1)
    padded_im[:, -pad_w:] = np.expand_dims(padded_im[:, -pad_w - 1], 1)
    return padded_im


def convolve(image: np.ndarray, kernel) -> np.ndarray:
    h, w = image.shape
    k_h, k_w = kernel.shape
    padded_image = get_padded_image(image, kernel.shape)
    conv_image = np.zeros(image.shape)
    for i in range(k_h):
        for j in range(k_w):
            conv_image += padded_image[i:i + h, j:j + w] * kernel[i, j]
    return conv_image


def get_distance_matrix(im_size):
    h, w = im_size
    indices = np.indices(im_size)
    D = (indices[0] - h // 2) ** 2 + (indices[1] - w // 2) ** 2
    return D


def get_gaussian_kernel(k=3, sigma=5):
    d = get_distance_matrix((k, k))
    G = 1/(2*np.pi* sigma**2) * np.exp(-d / (2 * sigma ** 2))  # Gaussian filter with kernel size k  and standard deviation sigma
    return G

