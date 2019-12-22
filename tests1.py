import os
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from math import sqrt, pow
import PIL.Image as Image
import random


def empty_rgb_matrix(lng, wid) -> list:
    return [[[0, 0, 0] for i in range(lng)] for j in range(wid)]


def empty_binary_matrix(lng, wid) -> list:
    return [[0 for i in range(lng)] for j in range(wid)]


def tint(tuple_one):
    res = []
    for el in tuple_one:
        res.append(int(el))
    return res


def tmul(tuple_one, mul):
    res = []
    for el in tuple_one:
        res.append(el * mul)
    return res


def tsum(tuple_one, tuple_two):
    res = []
    for el in range(len(tuple_one)):
        res.append(tuple_one[el] + tuple_two[el])
    return res


def tdiff(tuple_one, tuple_two):
    res = []
    for el in range(len(tuple_one)):
        res.append(tuple_one[el] - tuple_two[el])
    return res


def mean_of_mat(matrix: list) -> list:
    result = [0, 0, 0]
    t: list
    for t in matrix:
        p: list
        for p in t:
            for elm_num in range(len(p)):
                result[elm_num] += p[elm_num]
    for elm_num in range(len(matrix[0][0])):
        result[elm_num] = result[elm_num] // (len(matrix) * len(matrix[0]))
    return result


def cosinusian(x, mu, length):
    from math import cos
    return cos(pi / 2 * (abs(x - mu) / length))


def gaussian(x, mu, sig):
    return 1. / (sqrt(2. * pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)


def image_add_aperture(img: Image.Image, m_w, m_h) -> Image.Image:
    cast_img = Image.new(img.mode, (img.width + m_w, img.height + m_h))
    for i in range(img.width):
        for j in range(img.height):
            cast_img.putpixel((i + m_w, j + m_h), img.getpixel((i, j)))

    return cast_img


def image_expand(img: Image.Image, m_w, m_h) -> Image.Image:
    cast_img = Image.new(img.mode, (img.width + m_w, img.height + m_h))
    for i in range(img.width):
        for j in range(img.height):
            cast_img.putpixel((i + m_w // 2, j + m_h // 2), img.getpixel((i, j)))

    return cast_img


def mat_synth(frames: list) -> Image.Image:
    shot_h = len(frames[0])
    shot_w = len(frames)
    low_width = frames[0][0].width
    low_height = frames[0][0].height
    top_width = shot_w * low_width
    top_height = shot_h * low_height
    img = empty_rgb_matrix(top_width, top_height)

    for lwi in range(low_width):
        for lhi in range(low_height):
            for swi in range(shot_w):
                for shi in range(shot_h):
                    x_cor = lwi * shot_w + swi
                    y_cor = lhi * shot_h + shi
                    pix_a = frames[swi][shi].getpixel((lwi, lhi))
                    img[x_cor][y_cor] = pix_a

    out = Image.new("RGB", (len(img), len(img[0])))
    for i in range(len(img)):
        for j in range(len(img[0])):
            out.putpixel((i, j), tuple(img[i][j]))
    return out


def scan(scale, orig_img: Image.Image) -> list:
    print('size of original image [', orig_img.width, orig_img.height, ']')
    m_w = orig_img.width - scale
    m_h = orig_img.height - scale
    new_w = int(orig_img.width / scale)
    new_h = int(orig_img.height / scale)
    print('Output image size is [', new_w, new_h, ']')
    if not os.path.exists('results'):
        os.mkdir('results')

    frames = []

    for x in range(scale):
        j_frames = []
        for y in range(scale):
            new_img = orig_img.crop((x, y, m_w + x, m_h + y)).resize((new_w, new_h), Image.BILINEAR)
            new_img.resize((orig_img.width, orig_img.height), Image.NONE).save(
                'results/scan_yAxis' + str(y) + '_xAxis' + str(x) + '.bmp')
            j_frames = j_frames + [new_img]
        frames = frames + [j_frames]

    return frames


def scan2(scale, orig_img: Image.Image) -> list:
    cast_img = image_expand(orig_img, scale + 1, scale + 1)
    new_size_w = orig_img.width // scale
    new_size_h = orig_img.height // scale
    print('Output image size is [', new_size_w, new_size_h, ']')
    if not os.path.exists('results'):
        os.mkdir('results')
    half = scale // 2
    frames = []
    for x in range(scale):
        j_frames = []
        for y in range(scale):
            tmp_img = Image.new('RGB', (new_size_w, new_size_h))
            for i in range(new_size_w):
                isc = i * scale + x + half
                for j in range(new_size_h):
                    jsc = j * scale + y + half
                    mean = (0, 0, 0)
                    for m1 in range(0 - half, half, 1):
                        for m2 in range(0 - half, half, 1):
                            if 0 < isc + m1 < cast_img.width and 0 < jsc + m2 < cast_img.height:
                                mean = tsum(mean, cast_img.getpixel((isc + m1, jsc + m2)))
                    mean = tmul(mean, 1 / (scale * scale))
                    tmp_img.putpixel((i, j), tuple(tint(mean)))

            j_frames = j_frames + [tmp_img]
            tmp_img.resize((orig_img.width, orig_img.height), Image.NONE).save(
                'results/scan_yAxis' + str(y) + '_xAxis' + str(x) + '.bmp')
        frames = frames + [j_frames]

    return frames


def get_imp_noise(density: float, orig_img: Image.Image) -> Image.Image:
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


def tuptotal(tuple_one):
    su = 0
    for elm in tuple_one:
        su += elm
    return su


def get_max_and_min(image: Image.Image):
    max_val = (0, 0, 0)
    min_val = (255, 255, 255)

    for i in range(image.width):
        for j in range(image.height):
            if tuptotal(image.getpixel((i, j))) > tuptotal(max_val):
                max_val = image.getpixel((i, j))
            if tuptotal(image.getpixel((i, j))) < tuptotal(min_val):
                min_val = image.getpixel((i, j))

    return max_val, min_val


def get_bitmaps(noi_list: list) -> list:
    result = []
    image: Image.Image
    for image in noi_list:
        bitmap = empty_binary_matrix(image.width, image.height)
        upper, lower = (255, 255, 255), (0, 0, 0)  # get_max_and_min(image)
        for i in range(image.width):
            for j in range(image.height):
                pixel = image.getpixel((i, j))
                if tuptotal(lower) < tuptotal(pixel) < tuptotal(upper):
                    continue
                bitmap[i][j] = 1
        result += [bitmap]
    return result


def bitmap_sum(bitmap: list):
    summary = 0
    string: list
    for string in bitmap:
        for elm in string:
            summary += elm
    return summary


def get_smallest_bitmap(bitmaps: list):
    sums = []
    for bm in bitmaps:
        sums += [bitmap_sum(bm)]

    smallest = 0
    for i in range(len(sums)):
        if sums[i] < sums[smallest]:
            smallest = i

    return smallest, bitmaps[smallest]


def impulse_fusion_filter(img_list: list) -> Image.Image:
    while len(img_list) > 1:
        num, bm = get_smallest_bitmap(get_bitmaps(img_list))
        print('ESTIMATED ', len(img_list))
        best_sample: Image.Image = img_list.pop(num)
        best_sample.save('impulse/best_sample-' + str(len(img_list)) + '.bmp')
        for image in img_list:
            for i in range(image.width):
                for j in range(image.height):
                    if bm[i][j] == 0:
                        image.putpixel((i, j), best_sample.getpixel((i, j)))

    return img_list[0]


#
# def DCT(u,v, img: Image.Image):
#     from math import cos, pi
#     m = img.width
#     n = img.height
#     alpha = 1
#     beta = 1
#     if u == 0:
#         alpha = 1/sqrt(m)
#     elif u < m:
#         alpha = sqrt(2 / m)
#
#     if v == 0:
#         beta = 1 / sqrt(n)
#     elif v < n:
#         beta = sqrt(2/n)
#
#     res = (0,0,0)
#     for i in range(m):
#         for j in range(n):
#             cosin = cos(pi/m * (i+0.5) *u) * cos(pi/n * (j + 0.5)*v)
#             res = tsum(res, tmul(img.getpixel((i,j)), cosin))
#     tmul(res, alpha * beta)
#
#     return res


def dct2(block):
    from scipy.fftpack import dct
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    from scipy.fftpack import idct
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def toFreq(img: Image.Image) -> tuple:
    mat = empty_rgb_matrix(img.width, img.height)
    mat1 = np.zeros([img.width, img.height])
    mat2 = np.zeros([img.width, img.height])
    mat3 = np.zeros([img.width, img.height])

    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat[i][j] = img.getpixel((i, j))
            mat1[i][j] = mat[i][j][0]
            mat2[i][j] = mat[i][j][1]
            mat3[i][j] = mat[i][j][2]

    return dct2(mat1), dct2(mat2), dct2(mat3)


def fromFreq(mat1: np.ndarray, mat2: np.ndarray, mat3: np.ndarray):
    return idct2(mat1), idct2(mat2), idct2(mat3)


def get_image_from_channels(ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> Image.Image:
    size = (len(ch1), len(ch1[0]))
    newimg = Image.new('RGB', size)
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = [ch1[i][j], ch2[i][j], ch3[i][j]]
            newimg.putpixel((i, j), tuple(tint(pixel)))
    return newimg


def adjust_channels(coef: float, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:
    ch1_new = ch1 * coef
    ch2_new = ch2 * coef
    ch3_new = ch3 * coef
    return ch1_new, ch2_new, ch3_new


def sum_cha_packs(pack_one, pack_two):
    return pack_one[0] + pack_two[0], pack_one[1] + pack_two[1], pack_one[2] + pack_two[2]


def sum_two_images(img1: Image.Image, k1: float, img2: Image.Image, k2: float) -> Image.Image:
    cha_pack1 = adjust_channels(k1, *toFreq(img1))
    cha_pack2 = adjust_channels(k2, *toFreq(img2))
    return get_image_from_channels(*fromFreq(*sum_cha_packs(cha_pack1, cha_pack2)))


def tudaobratno(img: Image.Image) -> Image.Image:
    return get_image_from_channels(*fromFreq(*toFreq(img)))


image_name = 'rain'
second_image_name = 'sun-and-sky'

one = Image.open(image_name + '.bmp')
two = Image.open(second_image_name + '.bmp')

newimage = sum_two_images(one, 0.4, two, 0.6)

newimage.save(image_name + '_and_' + second_image_name + '_as_new_img.bmp')
# noise_set = get_noisy_set(0.9, 50, orig)
# print('GOT SET')
# sample: Image.Image = noise_set[0]
# sample.save('impulse/'+image_name+'_noise_sample.bmp')
# clear_image = impulse_fusion_filter(noise_set)
# print('FINISHED')
# clear_image.save('impulse/'+image_name + '_clear.bmp')

# t_frames = scan(16, orig)
# print('matsynth started')
# new_image = mat_synth(t_frames)
# new_image.save(image_name+'_matrix.bmp')
