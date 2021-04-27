import numpy as np
import matplotlib.pyplot as plt
import cv2
from restoration import gaussian_filter, adaptive_mmse, mmse, adaptive_shrink, unsharp_mask
from bm3d import bm3d, BM3DProfile, BM3DStages  # Installed using $pip install bm3d


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


def image_sharpening(input_im: np.ndarray, image: np.ndarray, plot=False):
    print('\n Image sharpening with varying gains: \n')
    K = np.arange(0, 1, 0.05)
    mse = np.zeros(K.shape)
    for i in range(K.shape[0]):  # sharpen image for different gains
        im_sharpened = unsharp_mask(input_im, k=K[i])
        mse[i] = np.mean((im_sharpened - image) ** 2)
        print(f'Gain K={K[i]}, MSE={mse[i]}')
        if plot:
            plt.imshow(im_sharpened, cmap='gray', vmin=0, vmax=255, interpolation=None)
            plt.title(f'Gain K={K[i]:.2f}, MSE={mse[i]:.2f}')
            plt.show()

    K_best = K[np.argmin(mse)]  # get the gain resulting in least MSE
    im_sharpened = unsharp_mask(input_im, k=K_best)

    print(f'Best K obtained for high boost filtering is {K_best:.2f}, MSE: {np.min(mse):.2f}')
    # Plot ref and output images, MSE vs K
    plt.subplot(131, title='Input denoised image')
    plt.imshow(input_im, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(132, title=f'Sharpened image K = {K_best:.2f}')
    plt.imshow(im_sharpened, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.subplot(133, title='Reference image')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation=None)
    plt.show()

    plt.plot(K, mse)
    plt.title('MSE Vs K')
    plt.xlabel('Gain K')
    plt.ylabel('MSE')
    plt.show()
    return


def bm3d_exp(dist='BM_Weiner'):
    image = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE)
    noise_sigmas = np.arange(5, 51, 5)  # vary sigma as 5, 10, ....,50

    mse_stage1 = np.zeros(noise_sigmas.shape)
    mse_stage2 = np.zeros(noise_sigmas.shape)
    mse_stage2_ht = np.zeros(noise_sigmas.shape)
    for i in range(len(noise_sigmas)):
        noise_sigma = noise_sigmas[i]
        noisy_im = image + np.random.normal(0, noise_sigma, image.shape)  # add gaussian noise
        noisy_im = np.clip(noisy_im, 0, 255)
        y = np.clip(image / 255, 0, 1)  # normalize pixels to range [0,1]
        noisy_y = np.clip(noisy_im / 255, 0, 1)
        print(f"Noise variance: {noise_sigma ** 2}")

        y_stage1 = bm3d(noisy_y, noise_sigma / 255,
                        stage_arg=BM3DStages.HARD_THRESHOLDING)  # get the denoised image after first stage of BM3D
        y_stage1 = np.clip(y_stage1, 0, 1)
        mse_stage1[i] = np.mean(((y_stage1 - y) * 255) ** 2)
        print("MSE after stage1 of BM3D", mse_stage1[i])

        y_stage2 = bm3d(noisy_y, noise_sigma / 255)  # denoised image after both the stages of BM3D
        y_stage2 = np.clip(y_stage2, 0, 1)
        mse_stage2[i] = np.mean(((y_stage2 - y) * 255) ** 2)
        print("MSE after stage2(Wiener) of BM3D", mse_stage2[i])

        # Use the stage 1 output to match blocks for the second stage
        if dist == 'BM_2Dtransform':  # Apply 2D transform and match blocks
            _, matches = bm3d(y_stage1, noise_sigma / 255, stage_arg=BM3DStages.HARD_THRESHOLDING,
                              blockmatches=(True, False))
            block_matches = matches[0]
        elif dist == 'BM_Weiner':  # Normalized l2 distance computed from basic estimate
            profile = BM3DProfile()
            profile.max_3d_size_wiener = 16
            _, matches = bm3d(noisy_y, noise_sigma / 255, profile, stage_arg=y_stage1, blockmatches=(False, True))
            block_matches = matches[1]

        y_stage2_ht = bm3d(noisy_y, noise_sigma / 255, stage_arg=BM3DStages.HARD_THRESHOLDING)
        y_stage2_ht = np.clip(y_stage2_ht, 0, 1)
        mse_stage2_ht[i] = np.mean(((y_stage2_ht - y) * 255) ** 2)
        print("MSE replacing stage2 of BM3D with HT", mse_stage2_ht[i])

        plt.figure(figsize=(15, 15))
        plt.subplot(121)
        plt.imshow(noisy_y, cmap='gray', vmin=0, vmax=1, interpolation=None)
        plt.title(f'Noisy image, sigma={noise_sigma}')

        plt.subplot(122)
        plt.imshow(y_stage1, cmap='gray', vmin=0, vmax=1, interpolation=None)
        plt.title(f'BM3D output after Stage1, mse={mse_stage1[i]:.2f}')
        plt.show()

        plt.figure(figsize=(15, 15))
        plt.subplot(121)
        plt.imshow(y_stage2, cmap='gray', vmin=0, vmax=1, interpolation=None)
        plt.title(f'BM3D output after Stage2, mse={mse_stage2[i]:.2f}')

        plt.subplot(122)
        plt.imshow(y_stage2_ht, cmap='gray', vmin=0, vmax=1, interpolation=None)
        plt.title(f'BM3D output with stage2 as HT, mse={mse_stage2_ht[i]:.2f}')
        plt.show()

    plt.plot(noise_sigmas, mse_stage1, label='MSE_stage1')
    plt.plot(noise_sigmas, mse_stage2, label='MSE_stage2')
    plt.plot(noise_sigmas, mse_stage2_ht, label='MSE_stage2_HT')
    plt.legend()
    plt.xticks(np.arange(0, 51, 5))
    plt.xlabel('noise sigma')
    plt.ylabel('MSE')

    plt.show()

    return [noise_sigmas, mse_stage1, mse_stage2, mse_stage2_ht]


def main():
    image_denoising()  # Variants of image denoising

    image = cv2.imread('lighthouse2.bmp', cv2.IMREAD_GRAYSCALE)  # Image sharpening
    noisy_im = image + np.random.normal(0, 10, image.shape)
    noisy_im = np.clip(noisy_im, 0, 255)
    filtered_im = gaussian_filter(noisy_im, k=7, sigma=1)
    image_sharpening(filtered_im, image, plot=True)

    bm3d_exp()  # BM3D analysis
    return


if __name__ == '__main__':
    main()
