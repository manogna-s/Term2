import os

from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
from bitstream import BitStream
from utils import dct2, idct2, get_padded_image, get_quantization_matrix
from sys import getsizeof
from copy import deepcopy

get_bin = lambda x, n: format(x, 'b').zfill(n)


class JPEG:
    def __init__(self, quant_matrix=None, img_path=None, bitstream=''):
        self.q_dct = None
        self.Q = quant_matrix
        self.code_dict = dict()
        self.code_dict[0] = '0'
        self.code_dict[-1] = '101'
        self.code_dict[1] = '100'
        if img_path is not None:
            self.input_im = imread(img_path)
        self.enc_bitstream = bitstream
        self.shift = False
        self.decoded_im = None
        self.im_shape = None

    def encode_dct_block(self, block):
        bitstream = ''
        for i in range(8):
            for j in range(8):
                n = block[i, j]
                if n in self.code_dict.keys():
                    code = self.code_dict[n]
                else:
                    K = int(np.log2(abs(n)))
                    category = get_bin(2 ** (K + 1) - 1, K)
                    s = abs(n) - (2 ** K)
                    sign_bit = '0' if n > 0 else '1'
                    code = category + '0' + sign_bit + get_bin(s, K)
                    self.code_dict[n] = code
                bitstream += code
        return bitstream

    def encoder(self):
        self.im_shape = self.input_im.shape
        image = self.input_im
        h, w = image.shape
        if h % 8 != 0 or w % 8 != 0:
            image = get_padded_image(self.input_im)
            self.im_shape = image.shape
        self.q_dct = np.zeros(self.im_shape)

        if self.shift:
            image = image.astype(int) - 128
        dct = np.zeros(image.shape)

        # Encoder
        code_dict = dict()
        code_dict[0] = '0'

        for i in range(h // 8):
            for j in range(w // 8):
                im_block = image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
                # dct[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = dct2(im_block)
                q_dct_block = np.floor(dct2(im_block) / self.Q + 0.5).astype(int)
                self.q_dct[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = q_dct_block
                code = self.encode_dct_block(q_dct_block)
                self.enc_bitstream += code
        return self.enc_bitstream

    def decode_block(self):
        dec_block = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                ind = self.enc_bitstream.index('0')
                dec_block[i, j] = self.code_dict[self.enc_bitstream[:2 * ind + 1]]
                self.enc_bitstream = self.enc_bitstream[2 * ind + 1:]
        return dec_block

    def decode(self):
        # Reconstruct
        self.decoded_im = np.zeros(self.im_shape)
        h, w = self.im_shape
        for i in range(h // 8):
            for j in range(w // 8):
                dec_block = self.decode_block()
                self.decoded_im[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = idct2(
                    dec_block * self.Q)

        if self.shift:
            self.decoded_im += 128

        self.decoded_im = np.uint8(np.clip(self.decoded_im, 0, 255))

        return self.decoded_im


def encode(block, code_dict):
    bitstream = ''

    print(np.unique(block), np.setdiff1d(np.unique(block), code_dict.keys()))
    for i in range(8):
        for j in range(8):
            n = block[i, j]
            if n in code_dict.keys():
                code = code_dict[n]
            else:
                K = int(np.log2(abs(n)))
                category = get_bin(2 ** (K + 1) - 1, K)
                print(type(category))
                s = abs(n) - (2 ** K)
                sign_bit = '0' if n > 0 else '1'
                code = category + '0' + sign_bit + get_bin(s, K)
                code_dict[n] = code
            bitstream += code
    return bitstream, code_dict


def jpeg(image: np.ndarray, Q=1, plot=False):
    h, w = image.shape
    if h % 8 != 0 or w % 8 != 0:
        image = get_padded_image(image)

    shift = False
    if shift:
        image = image.astype(int) - 128
    # print(np.min(image), np.max(image))
    dct = np.zeros(image.shape)
    q_dct = np.zeros(image.shape)
    image_reconstructed = np.zeros(image.shape)

    # Encoder
    code_dict = dict()
    code_dict[0] = '0'
    enc_bitstream = ''
    # enc_bitstream = BitStream()
    for i in range(h // 8):
        for j in range(w // 8):
            im_block = image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            # dct[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = dct2(im_block)
            q_dct_block = np.floor(dct2(im_block) / Q + 0.5).astype(np.int)
            q_dct[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = q_dct_block
            code, code_dict = encode(q_dct_block, code_dict.copy())
            # code = encode(q_dct_block, code_dict)
            # enc_bitstream.write([bool(int(char)) for char in code])
    print(len(enc_bitstream))

    # Reconstruct
    for i in range(h // 8):
        for j in range(w // 8):
            image_reconstructed[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = idct2(
                q_dct[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] * Q)

    if shift:
        image += 128
        image_reconstructed += 128
    image_reconstructed = np.clip(image_reconstructed, 0, 255)
    mse = np.mean((image - image_reconstructed) ** 2)
    print(f'MSE ={mse}')

    # print(np.max(dct), np.min(dct))
    # print(np.max(q_dct), np.min(q_dct))
    if plot:
        plt.figure()
        plt.imshow(dct, cmap='gray', vmax=np.max(dct) * 0.1, vmin=0)
        plt.title("8x8 DCTs of the image")
        plt.show()
        print(np.max(image_reconstructed), np.min(image_reconstructed))
        plt.imshow(image_reconstructed, cmap='gray', vmin=0, vmax=255)
        plt.show()

    return


def encode_no(n):
    # n = 511
    K = int(np.log2(abs(n)))
    category = get_bin(2 ** (K + 1) - 1, K)
    s = abs(n) - (2 ** K)
    sign_bit = '0' if n > 0 else '1'
    code = category + '0' + sign_bit + get_bin(s, K)
    # code_as_bools = [bool(int(char)) for char in code]
    # stream = BitStream()
    # stream.write(code_as_bools)
    # print(len(stream), getsizeof(stream))
    return code


def write_binary():
    binary_file = open("test.txt", "wb")
    binary_file.write(b'\x00')
    binary_file.close()
    import os
    file_length_in_bytes = os.path.getsize("test.txt")
    print(file_length_in_bytes)


def jpeg_encode_decode(img_path: str, out_path=None, quant_matrix=1, plot=False):
    input_image = imread('cameraman.tif')
    in_filesize = os.path.getsize('cameraman.tif') * 8
    print(f'Input file size in bits:{in_filesize}')

    encode_jpeg = JPEG(img_path=img_path, quant_matrix=quant_matrix)
    enc_bitstream = encode_jpeg.encoder()
    out_filesize = len(enc_bitstream)
    print(f'Output file size in bits: {out_filesize}')

    decode_jpeg = JPEG(bitstream=enc_bitstream, quant_matrix=quant_matrix)
    decode_jpeg.code_dict = dict((v, k) for k, v in encode_jpeg.code_dict.items())
    decode_jpeg.im_shape = encode_jpeg.im_shape
    decoded_im = decode_jpeg.decode()
    if out_path is not None:
        imsave(out_path, decoded_im)

    mse = np.mean((input_image - decoded_im) ** 2)
    comp_ratio = in_filesize / out_filesize
    print(f'MSE ={mse}')
    print(f'Compression ratio using given Q: {comp_ratio}\n\n')

    if plot:
        plt.subplot(1, 2, 1)
        plt.imshow(input_image, cmap='gray', vmin=0, vmax=255, interpolation=None)
        plt.title('Input image')
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_im, cmap='gray', vmin=0, vmax=255, interpolation=None)
        plt.title('Reconstructed image')
        plt.show()

    return mse, comp_ratio


def main():

    # Given quantization table Q
    Q = get_quantization_matrix(a=10, b=40, c=20)

    mse, comp_ratio = jpeg_encode_decode('cameraman.tif', out_path='reconstructed_im_Q.png', quant_matrix=Q, plot=True)

    # Compression by just rounding off DCT coefficients to nearest integer
    mse, comp_ratio = jpeg_encode_decode('cameraman.tif', out_path='rounding_off_DCT.png', quant_matrix=1, plot=True)



    return


if __name__ == '__main__':
    main()
