import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt


def draw_graph_and_save(graph_name: str, x_vector: list, y_vector: list, x_label: str, y_label: str):
    plt.plot(x_vector, y_vector)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(graph_name+'.png')


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


def gaussian(x, mu, sig):
    from math import sqrt, pi
    return 1. / (sqrt(2. * pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


def gauss_noisy_shots(amount: int, std: float, img_orig: Image.Image) -> list:

    shots = []
    for i in range(amount):
        noised_shot: Image.Image = enoise_image_mul(std, img_orig)
        shots.append(noised_shot)
    return shots


def image_add_aperture(img: Image.Image, m_w, m_h) -> Image.Image:
    cast_img = Image.new(img.mode, (img.width + m_w, img.height + m_h))
    for i in range(img.width):
        for j in range(img.height):
            cast_img.putpixel((i + m_w, j + m_h), img.getpixel((i, j)))

    return cast_img


def enoise_image_mul(std: float, img: Image.Image) -> Image.Image:
    new_img = img.copy()
    import random
    random.seed()
    for i in range(img.width):
        for j in range(img.height):
            pix = list(img.getpixel((i, j)))
            pix[0] = pix[0] * (1 + np.random.normal(0.0, std))
            pix[1] = pix[1] * (1 + np.random.normal(0.0, std))
            pix[2] = pix[2] * (1 + np.random.normal(0.0, std))
            new_img.putpixel((i, j), tuple(tint(pix)))
    return new_img


def pave_image(img: Image.Image, new_size) -> Image.Image:
    img_new = Image.new(img.mode, new_size)

    for i in range(img_new.width):
        for j in range(img_new.height):
            img_new.putpixel((i, j), img.getpixel((i % img.width, j % img.height)))
    return img_new


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
    for i in range(len(tuple_one)):
        tuple_one[i] *= mul
    return tuple_one


def tup_sum(tuple_one, tuple_two):
    res = []
    for i in range(len(tuple_one)):
        res.append(tuple_one[i] + tuple_two[i])
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
