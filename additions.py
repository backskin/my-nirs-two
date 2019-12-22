import PIL.Image as Image
import numpy as np


def adjust_channels(coef: float, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:
    return ch1 * coef, ch2 * coef, ch3 * coef


def dct2(block):
    from scipy.fftpack import dct
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    from scipy.fftpack import idct
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def low_pass_filter(border: int, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:
    len_bord = border if len(ch1) > border > 0 else len(ch1) - 1
    wid_bord = border if len(ch1[0]) > border > 0 else len(ch1[0]) - 1
    ch1_new = ch1
    ch1_new[len_bord:, wid_bord:] = 0
    ch2_new = ch2
    ch2_new[len_bord:, wid_bord:] = 0
    ch3_new = ch3
    ch3_new[len_bord:, wid_bord:] = 0
    return ch1_new, ch2_new, ch3_new


def high_pass_filter(border: int, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:
    len_bord = border if len(ch1) > border > 0 else 0
    wid_bord = border if len(ch1[0]) > border > 0 else 0
    ch1_new = ch1
    ch1_new[:len_bord, :wid_bord] = 0
    ch2_new = ch2
    ch2_new[:len_bord, :wid_bord] = 0
    ch3_new = ch3
    ch3_new[:len_bord, :wid_bord] = 0
    return ch1_new, ch2_new, ch3_new


def get_imp_noise(density: float, orig_img: Image.Image) -> Image.Image:
    import random
    random.seed()
    image = orig_img.copy()
    for i in range(image.width):
        for j in range(image.height):
            if random.random() <= density:
                if random.random() > 0.5:
                    image.putpixel((i, j), (255, 255, 255))
                else:
                    image.putpixel((i, j), (0, 0, 0))
    return image


def get_random_noisy_set(orig_img: Image.Image, amount: int) -> list:
    import random
    out = []
    for k in range(amount):
        density = 0.01 + random.random() / 4
        out += [get_imp_noise(density, orig_img)]
    return out


def get_noisy_set(density: float, amount: int, orig_img: Image.Image) -> list:
    out = []
    for k in range(amount):
        out += [get_imp_noise(density, orig_img)]
    return out


def gaussian(x, mu, sig):
    from math import sqrt, pi
    return 1. / (sqrt(2. * pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


def image_add_aperture(img: Image.Image, m_w, m_h) -> Image.Image:
    cast_img = Image.new(img.mode, (img.width + m_w, img.height + m_h))
    for i in range(img.width):
        for j in range(img.height):
            cast_img.putpixel((i + m_w, j + m_h), img.getpixel((i, j)))

    return cast_img


def enoise_image_mul(std: float, img: Image.Image) -> Image.Image:
    new_img = img.copy()
    for i in range(img.width):
        for j in range(img.height):
            old_pix = img.getpixel((i, j))
            new_img.putpixel((i, j), tuple(tint(tup_sum(old_pix, tup_mul(old_pix, np.random.normal(0.0, std))))))
    return new_img


def enoise_image_mul_rgb(std: float, img: Image.Image) -> Image.Image:
    new_img = img.copy()
    for i in range(img.width):
        for j in range(img.height):
            old_pix = img.getpixel((i, j))
            noise_pix = []
            for k in range(len(old_pix)):
                noise_pix.append(old_pix[k] * np.random.normal(0.0, std))
            new_img.putpixel((i, j), tuple(tint(tup_sum(old_pix, noise_pix))))
    return new_img


def pave_image(img: Image.Image, new_size) -> Image.Image:
    newimg = Image.new(img.mode, new_size)
    for i in range(newimg.width):
        for j in range(newimg.height):
            newimg.putpixel((i, j), img.getpixel((divmod(i, img.width)[1], divmod(j, img.height)[1])))
    return newimg


def expand_image(img: Image.Image, m_w, m_h) -> Image.Image:
    cast_img = Image.new(img.mode, (img.width + m_w, img.height + m_h))
    for i in range(img.width):
        for j in range(img.height):
            cast_img.putpixel((i + m_w // 2, j + m_h // 2), img.getpixel((i, j)))

    return cast_img


def empty_rgb_matrix(lng, wid) -> list:
    return [[[0, 0, 0] for i in range(lng)] for j in range(wid)]


def tint(tuple_one):
    res = []
    for el in tuple_one:
        res.append(int(el))
    return res


def tup_mul(tuple_one, mul):
    res = []
    for el in tuple_one:
        res.append(el * mul)
    return res


def tup_sum(tuple_one, tuple_two):
    res = []
    for el in range(len(tuple_one)):
        res.append(tuple_one[el] + tuple_two[el])
    return res


def tup_dif(tuple_one, tuple_two):
    res = []
    for el in range(len(tuple_one)):
        res.append(tuple_one[el] - tuple_two[el])
    return res


def tup_total(tuple_one):
    total = 0
    for elm in tuple_one:
        total += elm
    return total


def tup_max(tuple_list: list):
    index = 0
    max_amount = 0
    for t_index in range(len(tuple_list)):
        total = tup_total(tuple_list[t_index])
        if total > max_amount:
            index = t_index
            max_amount = total
    return index


def tup_sort(tuple_list: list):
    result = []
    while len(tuple_list) > 0:
        ind = tup_max(tuple_list)
        result.append(tuple_list.pop(ind))
    return result


def tup_median(tuple_list: list):
    return tuple_list[len(tuple_list) // 2]
