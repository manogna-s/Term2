from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import os


def get_padded_image(bin_image: np.ndarray) -> np.ndarray:
    h, w = bin_image.shape
    pad_h, pad_w = h % 8, w % 8
    padded_im = np.ones((h + pad_h, w + pad_w))
    padded_im[:h, :w] = bin_image
    # pad rows at the bottom
    padded_im[-pad_h:] = padded_im[-pad_h - 1]
    # pad columns at the right end
    padded_im[:, -pad_w:] = np.expand_dims(padded_im[:, -pad_w - 1], 1)
    return padded_im


def dct2(block):
    return fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(block):
    return fftpack.idct(fftpack.idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def get_quantization_matrix(a=10, b=40, c=20):
    Q = np.zeros((8, 8))
    Q[:, :] = b
    Q[:4, :4] = a
    Q[0, 0] = c
    return Q


get_bin = lambda x, n: format(x, 'b').zfill(n)      # get binary representation of given x


class JPEG:
    def __init__(self, quant_matrix=None, img_path=None, bitstream=''):
        # self.q_dct = None
        self.Q = quant_matrix
        self.code_dict = {0: '0', -1: '101', 1: '100'}
        if img_path is not None:
            self.input_im = imread(img_path)
        self.enc_bitstream = bitstream
        self.decoded_im = None
        self.im_shape = None

    def encode_dct_block(self, block):
        bitstream = ''
        for i in range(8):
            for j in range(8):
                x = block[i, j]
                if x in self.code_dict.keys():  # Get code from the dictionary if it exists
                    code = self.code_dict[x]
                else:  # Identify category and code, store it in the dict for further use.
                    K = int(np.log2(abs(x)))                            # e.g., x=-5, K = 2
                    category = get_bin(2 ** (K + 1) - 1, K)             # get code for the category get_bin(7)='111'
                    s = abs(x) - (2 ** K)                               # get code for +5/-5, s = 1, get_bin(s,2) = '01'
                    sign_bit = '0' if x > 0 else '1'                    # sign bit = '1' for x=-5
                    code = category + '0' + sign_bit + get_bin(s, K)    # code = '111'+'0'+'1'+'01'
                    self.code_dict[x] = code
                bitstream += code
        return bitstream

    def encoder(self):
        self.im_shape = self.input_im.shape
        image = self.input_im
        h, w = image.shape
        if h % 8 != 0 or w % 8 != 0:
            image = get_padded_image(self.input_im)
            self.im_shape = image.shape

        for i in range(h // 8):
            for j in range(w // 8):
                im_block = image[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]  # Get 8x8 block
                # Get DCT for the block and quantize with the given quantization matrix Q. \
                # If Q=1, then it just rounds off the DCT coefficients to the nearest integer
                q_dct_block = np.floor(dct2(im_block) / self.Q + 0.5).astype(int)
                code = self.encode_dct_block(q_dct_block)  # Encode the quantized DCT block
                self.enc_bitstream += code  # concatenate bitstream of each block
        return self.enc_bitstream

    def decode_block(self):
        dec_block = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                # eg., code: '11001' belongs to category '110', get index of first '0' to  get identify category \
                # This also indicates the length of the code 2*index+1
                index = self.enc_bitstream.index('0')
                # Get the code and decode from the dictionary
                dec_block[i, j] = self.code_dict[self.enc_bitstream[:2 * index + 1]]
                # Remove the decoded string from the bitstream
                self.enc_bitstream = self.enc_bitstream[2 * index + 1:]
        return dec_block

    def decode(self):
        # Reconstruct image
        self.decoded_im = np.zeros(self.im_shape)
        h, w = self.im_shape
        for i in range(h // 8):
            for j in range(w // 8):
                dec_block = self.decode_block()  # decode one block from the encoded binary string
                # Reconstruct one block taking inverse DCT
                self.decoded_im[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = idct2(dec_block * self.Q)

        self.decoded_im = np.uint8(np.clip(self.decoded_im, 0, 255))  # Clip values to valid range [0,255]
        return self.decoded_im


def jpeg_encode_decode(img_path: str, out_path=None, quant_matrix=1, plot=False):
    input_image = imread(img_path)
    in_filesize = os.path.getsize(img_path) * 8  # Get Input image file size in bits
    print(f'Input file size in bits:{in_filesize}')

    '''
    Encoder: 
    Input: Image file path, quantization matrix
    Returns: Encoded bitstream
    '''
    encode_jpeg = JPEG(img_path=img_path, quant_matrix=quant_matrix)  # Create JPEG instance to encode image
    enc_bitstream = encode_jpeg.encoder()  # Get encoded bitstream
    out_filesize = len(enc_bitstream)  # output file size in bits = length of the bitstream
    print(f'Output file size in bits: {out_filesize}')

    '''
    Decoder: 
    Input: Encoded bitstream, image shape
    Params needed to decode: quantization matrix and code dictionary used while encoding
    Returns: Reconstructed image
    '''
    decode_jpeg = JPEG(bitstream=enc_bitstream, quant_matrix=quant_matrix)  # Create JPEG instance to decode image
    # Reverse the encoder dict to map binary strings to dct coefficient as decoder dict
    decode_jpeg.code_dict = dict((v, k) for k, v in encode_jpeg.code_dict.items())
    decode_jpeg.im_shape = encode_jpeg.im_shape
    decoded_im = decode_jpeg.decode()
    if out_path is not None:
        imsave(out_path, decoded_im)

    # Compute MSE and compression ratio
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
        plt.clf()

    return mse, comp_ratio


def find_best_Q(img_path, thresh):
    # Find best a,b,c such that compression ratio is more than the given ref thresh
    best_mse = 10000
    params = np.arange(10, 101, 10)
    best_params = []

    # Search for best a,b,c
    for a in params:
        for b in params:
            for c in params:
                Q = get_quantization_matrix(a=a, b=b, c=c)
                mse_q, comp_ratio_q = jpeg_encode_decode('cameraman.tif', quant_matrix=Q)
                if comp_ratio_q >= thresh:  # check if compression ratio is better than the given ref
                    if mse_q < best_mse:  # check and update least mse, params
                        best_params = [a, b, c]
                        best_mse = mse_q
    return best_params
