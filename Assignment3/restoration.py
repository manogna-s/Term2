import numpy as np
from utils import get_gaussian_kernel, convolve


def unsharp_mask(image: np.ndarray, k: float) -> np.ndarray:
    hpf = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    hp_image = convolve(image, hpf)                         # Apply the high pass filter on the image
    sharpened_im = image + k * hp_image                     # Add the high high pass component scaled with the given gain
    sharpened_im = np.clip(sharpened_im, 0, 255)
    return sharpened_im


def gaussian_filter(image: np.ndarray, k=5, sigma=5):
    gaussian_kernel = get_gaussian_kernel(k=k, sigma=sigma)
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)         #normalize so that filter coefficients sum to 1
    filtered_im = convolve(image, gaussian_kernel)
    filtered_im = np.clip(filtered_im, 0, 255)
    return np.uint8(filtered_im)


def mmse(image, noise_var=100, k_lpf=5, sigma_lpf=1):
    lpf = get_gaussian_kernel(k=k_lpf, sigma=sigma_lpf)

    # estimate noise variance in high pass signal
    hpf = -lpf
    hpf[k_lpf // 2, k_lpf // 2] += 1
    var_z1 = np.sum(hpf ** 2) * noise_var

    y = image
    y_lp = convolve(y, lpf)
    y1 = y - y_lp
    var_y1 = np.mean(y1 ** 2)
    var_x1 = var_y1 - var_z1
    filtered_im = y_lp + (var_x1 / var_y1) * y1
    return np.uint8(filtered_im)


def adaptive_mmse(image: np.ndarray, k=32, s=16, k_lpf=5, sigma_lpf=1, noise_var=100):
    lpf = get_gaussian_kernel(k=k_lpf, sigma=sigma_lpf)

    # estimate noise variance in high pass signal
    hpf = -lpf
    hpf[k_lpf // 2, k_lpf // 2] += 1
    var_z1 = np.sum(hpf ** 2) * noise_var
    h, w = image.shape

    filtered_im = np.zeros(image.shape)
    avg_count = np.zeros(image.shape)

    for i in range(h // s - 1):
        for j in range(w // s - 1):
            y = image[i * s:i * s + k, j * s:j * s + k]
            y_lp = convolve(y, lpf)
            y1 = y - y_lp
            var_y1 = np.mean(y1 ** 2)
            var_x1 = var_y1 - var_z1
            filtered_im[i * s:i * s + k, j * s:j * s + k] += y_lp + (var_x1 / var_y1) * y1
            avg_count[i * s:i * s + k, j * s:j * s + k] += 1
    filtered_im = filtered_im / avg_count
    return np.uint8(filtered_im)


def sureshrink_thresh(y1, var_z1):
    h, w = y1.shape
    th = np.unique(np.abs(y1))
    sure = np.zeros(th.shape)
    for i in range(th.shape[0]):  # iterate over all possible thresholds and compute SURE
        t = th[i]
        sure[i] = h * w * var_z1 + np.sum(np.minimum(np.abs(y1), t) ** 2) - 2 * var_z1 * np.sum(np.abs(y1) < t)
    return th[np.argmin(sure)]


def adaptive_shrink(image: np.ndarray, k=32, s=32, k_lpf=5, sigma_lpf=1, noise_var=100, sure=True):
    lpf = get_gaussian_kernel(k=k_lpf, sigma=sigma_lpf)

    # estimate noise variance in high pass signal
    hpf = -lpf
    hpf[k_lpf // 2, k_lpf // 2] += 1
    var_z1 = np.sum(hpf ** 2) * noise_var
    h, w = image.shape

    filtered_im = np.zeros(image.shape)

    for i in range(h // s):                                                         # stride of 32
        for j in range(w // s):
            y = image[i * s:i * s + k, j * s:j * s + k]                             # get 32x32 patch
            y_lp = convolve(y, lpf)
            y1 = y - y_lp
            if sure:                                                                # get threshold by minimising SURE
                thresh = sureshrink_thresh(y1, var_z1)
            else:                                                                   # threshold from estimated high pass signal and noise variance
                var_y1 = np.mean(y1 ** 2)
                var_x1 = var_y1 - var_z1
                thresh = np.sqrt(2) * var_z1 / np.sqrt(var_x1)
            x1 = np.zeros(y1.shape)                                                 # get shrinkage estimate for high pass coefficients
            x1[np.where(y1 > thresh)] = y1[np.where(y1 > thresh)] - thresh
            x1[np.where(y1 < -thresh)] = y1[np.where(y1 < -thresh)] + thresh

            filtered_im[i * s:i * s + k, j * s:j * s + k] += y_lp + x1              # update estimates

    return np.uint8(filtered_im)



def adaptive_shrink_overlap(image: np.ndarray, k=32, s=16, k_lpf = 5, sigma_lpf=1, noise_var=100, sure=True):   #adaptive shrink with overlap
    lpf = get_gaussian_kernel(k=k_lpf, sigma=sigma_lpf)

    # estimate noise variance in high pass signal
    hpf = -lpf
    hpf[k_lpf//2, k_lpf//2] += 1
    var_z1 = np.sum(hpf ** 2) * noise_var
    h, w = image.shape

    filtered_im = np.zeros(image.shape)
    avg_count = np.zeros(image.shape)

    for i in range(h // s - 1):                                 #stride of 16
        for j in range(w // s - 1):
            y = image[i * s:i * s + k, j * s:j * s + k]         #get 32x32 patch
            y_lp = convolve(y, lpf)
            y1 = y - y_lp
            if sure:                                            #get threshold by minimising SURE
                thresh = sureshrink_thresh(y1, var_z1)
            else:                                               #threshold from estimated high pass signal and noise variance
                var_y1 = np.mean(y1 ** 2)
                var_x1 = var_y1 - var_z1
                thresh = np.sqrt(2)*var_z1/np.sqrt(var_x1)

            x1 = np.zeros(y1.shape)                             #get shrinkage estimate for high pass coefficients
            x1[np.where(y1 > thresh)] = y1[np.where(y1 > thresh)] - thresh
            x1[np.where(y1 < -thresh)] = y1[np.where(y1 < -thresh)] + thresh

            filtered_im[i * s:i * s + k, j * s:j * s + k] += y_lp + x1         #update estimates
            avg_count[i * s:i * s + k, j * s:j * s + k] += 1                   #keep track of the no. of estimates to average in the overlapping regions

    filtered_im = filtered_im / avg_count                       #average the estimates in overlapping regions
    return np.uint8(filtered_im)
