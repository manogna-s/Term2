from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack


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


def dct2(a):
    return fftpack.dct( fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )


def idct2(a):
    return fftpack.idct( fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def get_quantization_matrix(a=10, b=40, c=20):
    Q = np.zeros((8, 8))
    Q[:, :] = b
    Q[:4, :4] = a
    Q[0, 0] = c
    return  Q