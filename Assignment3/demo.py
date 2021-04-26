import numpy as np
import matplotlib.pyplot as plt
import cv2
from restoration import gaussian_filter, adaptive_mmse, mmse, adaptive_shrink, highboost_filter
from bm3d import bm3d, BM3DProfile
from PIL import Image


def image_denoising():
    # Generate noisy image
    image = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE)
    noisy_im = image + np.random.normal(0, 10, image.shape)
    noisy_im = np.clip(noisy_im, 0, 255)
    noisy_im = np.uint8(noisy_im)

    noisy_mse = np.mean((noisy_im.astype(float) - image.astype(float)) ** 2)
    print(f'MSE of noisy image:{noisy_mse}')

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.title('Clean image')
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_im, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.title('Noisy image')
    plt.show()

    # Apply low pass gaussian filter of varying filter lengths and standard deviations
    kernels = [3, 7, 11]
    for sigma in [0.1, 1, 2, 4, 8]:
        for i in range(3):
            filtered_im = gaussian_filter(noisy_im, k=kernels[i], sigma=sigma)
            mse = np.mean((filtered_im.astype(float) - image.astype(float)) ** 2)
            plt.subplot(1, 3, i + 1)
            plt.imshow(filtered_im, cmap='gray', vmin=0, vmax=255, interpolation=None)
            plt.title(f'Kernel size: {kernels[i]}, sigma: {sigma}, mse: {mse:.2f}')
            print(f'Kernel size: {kernels[i]}, sigma: {sigma}, mse: {mse}')
        plt.show()


    # Denoise image using MMSE filter
    k_lpf = 7
    sigma_lpf = 1
    out = mmse(noisy_im, k_lpf=k_lpf, sigma_lpf=sigma_lpf)
    mmse_mse = np.mean((out.astype(float) - image.astype(float)) ** 2)
    plt.imshow(out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.title(f'MMSE (LPF: k={k_lpf}, sigma = {sigma_lpf}), mse: {mmse_mse:.2f}')
    plt.show()

    # Denoise image using Adaptive MMSE filter
    adapt_mmse_out = adaptive_mmse(noisy_im, k_lpf=k_lpf, sigma_lpf=sigma_lpf)
    adapt_mmse_mse = np.mean((adapt_mmse_out.astype(float) - image.astype(float)) ** 2)
    plt.imshow(adapt_mmse_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.title(f'Adaptive MMSE (LPF: k={k_lpf}, sigma = {sigma_lpf}), MSE: {adapt_mmse_mse:.2f} ')
    plt.show()

    # Denoise image using Adaptive Sureshrink estimator
    adapt_sureshrink_out = adaptive_shrink(noisy_im, k_lpf=k_lpf, sigma_lpf=sigma_lpf)
    adapt_sureshrink_mse = np.mean((adapt_sureshrink_out.astype(float) - image.astype(float)) ** 2)
    plt.imshow(adapt_sureshrink_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.title(f'Adaptive SureShrink (LPF: k={k_lpf}, sigma = {sigma_lpf}), mse: {adapt_sureshrink_mse:.2f}')
    plt.show()

    # Denoise image using Adaptive Shrinkage estimator
    adapt_shrink_out = adaptive_shrink(noisy_im, k_lpf=k_lpf, sigma_lpf=sigma_lpf, sure=False)
    adapt_shrink_mse = np.mean((adapt_shrink_out.astype(float) - image.astype(float)) ** 2)
    plt.imshow(adapt_shrink_out, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.title(f'Adaptive Shrink (LPF: k={k_lpf}, sigma = {sigma_lpf}), mse: {adapt_shrink_mse:.2f}')
    plt.show()


def image_sharpening(input_im: np.ndarray, image: np.ndarray):
    K = np.arange(0, 1, 0.05)
    mse = np.zeros(K.shape)
    for i in range(K.shape[0]):
        im_highboost = highboost_filter(input_im, k=K[i])
        mse[i] = np.mean((im_highboost - image) ** 2)
        print(K[i], mse[i])
    plt.imshow(im_highboost, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    K_best = K[np.argmin(mse)]
    im_highboost = highboost_filter(input_im, k=K_best)

    print(f'Best K obtained for high boost filtering is {K_best:.2f}, MSE: {np.min(mse):.2f}')
    # Plot ref and output images, MSE vs K
    plt.subplot(131, title='Input denoised image')
    plt.imshow(input_im, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(132, title=f'Highboost filtered image K = {K_best:.2f}')
    plt.imshow(im_highboost, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(133, title='Reference image')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.plot(K, mse)
    plt.title('MSE Vs K')
    plt.show()
    return


# def bm3d_exp():
#     image = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE)
#     noisy_im = image + np.random.normal(0, 10, image.shape)
#     noisy_im = np.clip(noisy_im, 0, 255) / 255
#     noisy_im = Image.fromarray(noisy_im)
#     out = bm3d(noisy_im, 10 / 255)
#     plt.subplot(121, title='Input noisy image')
#     plt.imshow(noisy_im, cmap='gray', vmin=0, vmax=255, interpolation=None)
#     plt.subplot(122, title=f'Highboost filtered image K = {K_best:.2f}')
#     plt.imshow(out, cmap='gray', vmin=0, vmax=255, interpolation=None)


def main():
    image = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE)
    noisy_im = image + np.random.normal(0, 10, image.shape)
    noisy_im = np.clip(noisy_im, 0, 255)

    filtered_im = gaussian_filter(noisy_im, k=3, sigma=1)
    #image_denoising()
    sharpened_im = image_sharpening(filtered_im, image)
    # bm3d_exp()

    return


if __name__ == '__main__':
    main()
