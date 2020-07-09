import os
import numpy as np
import cv2

from settings import ALLOWED_EXTS


def process_normalization(image):
    """
    Normalize an image.
    :param image: frame with rgb.
    :return:
    """

    h, w = image.shape[-3:-1]

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255

    mag, ang = cv2.cartToPolar(image[..., 0], image[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    processed_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.stack(processed_image)


def process_histogram_equalization(img_in):
    """
    Equalize histogram of an image.
    :param img_in: frame with rgb.
    :return:
    """

    # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])

    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype('uint8')

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))

    return img_out


def is_allowed(path):

    return os.path.splitext(path)[1] in ALLOWED_EXTS


def load_image(path, is_rotate=False, is_norm=False, is_equ=False):

    if is_allowed(path):
        img = cv2.imread(path)

        if is_rotate:
            pass

        if is_norm:
            img = process_normalization(image=img)

        if is_equ:
            img = process_histogram_equalization(img_in=img)

        return img

    else:
        return None


if __name__ == '__main__':

    pass
