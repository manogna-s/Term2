import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy import stats
from skimage import img_as_float
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
import lpips
from utils import get_quantization_matrix, jpeg_encode_decode, find_best_Q


def problem1():
    # Compression using given quantization table Q
    Q = get_quantization_matrix(a=10, b=40, c=20)
    _, comp_ratio_ref = jpeg_encode_decode('cameraman.tif', out_path='reconstructed_im_Q.png', quant_matrix=Q,
                                           plot=False)

    # Compression by just rounding off DCT coefficients to nearest integer
    _, comp_ratio_dct = jpeg_encode_decode('cameraman.tif', out_path='rounding_off_DCT.png', quant_matrix=1, plot=False)

    # Get best a,b,c in Q that minimises MSE with least filesize(or best compression ratio)
    best_params = find_best_Q('cameraman.tif', thresh=comp_ratio_ref)
    best_a, best_b, best_c = best_params
    print(f'best_params: [a,b,c] = {[best_a, best_b, best_c]}')
    best_Q = get_quantization_matrix(a=best_a, b=best_b, c=best_c)
    _, best_comp_ratio = jpeg_encode_decode('cameraman.tif', out_path='reconstructed_im_bestQ.png', quant_matrix=best_Q,
                                            plot=True)
    return


def problem2():
    mat = scipy.io.loadmat('hw5.mat')
    blur_orgs = mat['blur_orgs'][0]
    valid_idxs = np.where(blur_orgs == 0)  # get blur image indices. 145 images
    blur_dmos = mat['blur_dmos'][0][valid_idxs]
    refnames_blur = mat['refnames_blur'][0][valid_idxs]  # original image names for 145 images

    n_files = refnames_blur.shape[0]  # compute metrics over 145 images
    df = pd.DataFrame(columns=['Blur img', 'Ref img', 'DMOS', 'MSE', 'SSIM', 'LPIPS'])  # save all metrics

    loss_fn = lpips.LPIPS(net='alex', spatial=False)  # load alexnet model to get lpips score

    for i in range(n_files):
        blur_img_name = 'img' + str(i + 1) + '.bmp'
        ref_img_name = refnames_blur[i][0]
        blur_img_path = 'gblur/' + blur_img_name
        ref_img_path = 'refimgs/' + ref_img_name

        blur_img = img_as_float(imread(blur_img_path))  # convert pixel values to range [0,1]
        ref_img = img_as_float(imread(ref_img_path))

        mse_im = np.mean((blur_img - ref_img) ** 2)  # compute MSE between reference and blur image
        ssim_im = ssim(blur_img, ref_img, multichannel=True)  # compute ssim

        # load input images and normalizes to [-1,1] as required for lpips
        lpips_ref_im = lpips.im2tensor(lpips.load_image(ref_img_path))
        lpips_blur_im = lpips.im2tensor(lpips.load_image(blur_img_path))
        lpips_im = np.squeeze(loss_fn.forward(lpips_blur_im, lpips_ref_im).detach().numpy())  # compute lpips measure

        if i == 0:  # Plot MSE and SSIM map for an image
            mse_map = np.uint8(np.sum((blur_img - ref_img) ** 2, axis=2) * 255)
            ssim_im, ssim_map = ssim(blur_img, ref_img, multichannel=True, full=True)
            plt.imshow(mse_map, cmap='gray', vmin=np.min(mse_map), vmax=np.max(mse_map))
            plt.title('MSE map')
            plt.show()
            plt.imshow(ssim_map, cmap='gray')
            plt.title('SSIM map')
            plt.show()

        df.loc[str(i + 1)] = [blur_img_name, ref_img_name, blur_dmos[i], mse_im, ssim_im, lpips_im]

    df.to_csv('Quality_metrics.csv')

    # Compute Spearman rank order correlation coefficient
    dmos_mse_corr, _ = stats.spearmanr(blur_dmos, df['MSE'])
    dmos_ssim_corr, _ = stats.spearmanr(blur_dmos, df['SSIM'])
    dmos_lpips_corr, _ = stats.spearmanr(blur_dmos, df['LPIPS'])
    print('Spearman Rank order correlation coefficient of the following pairs of scores: ')
    print(f'DMOS and MSE: {dmos_mse_corr}')
    print(f'DMOS and SSIM: {dmos_ssim_corr}')
    print(f'DMOS and LPIPS: {dmos_lpips_corr}')

    return


def main():
    problem1()
    problem2()
    return


if __name__ == '__main__':
    main()
