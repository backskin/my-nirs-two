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
            new_img = orig_img.crop((x, y, m_w + x, m_h + y)).resize((new_w, new_h), Image.LINEAR)
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


def expand_img(img:Image.Image, new_size) -> Image.Image:
    newimg = Image.new(img.mode, new_size)
    for i in range(newimg.width):
        for j in range(newimg.height):
            newimg.putpixel((i,j), img.getpixel((divmod(i, img.width)[1], divmod(j, img.height)[1])))
    return newimg


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


def low_pass_filter(border: int, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:
    l = border if len(ch1) > border > 0 else len(ch1)-1
    w = border if len(ch1[0]) > border > 0 else len(ch1[0])-1
    ch1_new = ch1
    ch1_new[l:, w:] = 0
    ch2_new = ch2
    ch2_new[l:, w:] = 0
    ch3_new = ch3
    ch3_new[l:, w:] = 0
    return ch1_new, ch2_new, ch3_new


def high_pass_filter(border: int, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:
    l = border if len(ch1) > border > 0 else 0
    w = border if len(ch1[0]) > border > 0 else 0
    ch1_new = ch1
    ch1_new[:l, :w] = 0
    ch2_new = ch2
    ch2_new[:l, :w] = 0
    ch3_new = ch3
    ch3_new[:l, :w] = 0
    return ch1_new, ch2_new, ch3_new


def adjust_channels(coef: float, ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> tuple:

    ch1_new = ch1 * coef
    ch2_new = ch2 * coef
    ch3_new = ch3 * coef

    return ch1_new, ch2_new, ch3_new


def sum_cha_packs(pack_one, pack_two):
    return pack_one[0] + pack_two[0], pack_one[1] + pack_two[1], pack_one[2] + pack_two[2]


def fuse_two_images(img1: Image.Image, b1: int, c1: int, k1: float, img2: Image.Image, b2: int, c2: int, k2: float) -> Image.Image:
    cha_pack1 = adjust_channels(k1, *toFreq(img1))
    cha_pack1 = low_pass_filter(b1, *cha_pack1)
    cha_pack1 = high_pass_filter(c1, *cha_pack1)
    cha_pack2 = adjust_channels(k2, *toFreq(img2))
    cha_pack2 = low_pass_filter(b2, *cha_pack2)
    cha_pack2 = high_pass_filter(c2, *cha_pack2)
    return get_image_from_channels(*fromFreq(*sum_cha_packs(cha_pack1, cha_pack2)))


def tudaobratno(img: Image.Image) -> Image.Image:
    return get_image_from_channels(*fromFreq(*toFreq(img)))



def embed_wmark_additive(alpha: float, img_orig: Image.Image, watermark: Image.Image) -> Image.Image:
    newimg = img_orig.copy()
    wm_expanded = expand_img(watermark, img_orig.size)
    for i in range(img_orig.width):
        for j in range(img_orig.height):
            orig_pixel = img_orig.getpixel((i, j))
            pelmeni = 1
            mean = 0
            for el in orig_pixel:
                mean += el
            mean /= len(orig_pixel)
            if mean < 128:
                pelmeni = -1
            newimg.putpixel((i,j), tsum(orig_pixel, tmul(wm_expanded.getpixel((i, j)), alpha*pelmeni)))

    return newimg


def quantum(channels, delta):
    output = []
    for el in channels:
        output.append(delta * int((el+0.5) / delta))
    return output


def generate_dither_vectors(key, delta, size):
    random.seed(key)
    d0 = np.zeros([*size, 3])
    d1 = np.zeros([*size, 3])
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(3):
                r = random.random()
                d0[i][j][k] = int(delta*(r-0.5))
                d1[i][j][k] = int(d0[i][j][k] - np.sign(d0[i][j][k])*(delta/2))

    return d0, d1


def embed_wmark_dm_qim(key, delta, img_orig: Image.Image, wmark: Image.Image) -> Image.Image:
    newimg = img_orig.copy()
    d0, d1 = generate_dither_vectors(key, delta, img_orig.size)
    for i in range(img_orig.width):
        for j in range(img_orig.height):
            w_x, w_y = divmod(i, wmark.width)[1], divmod(j, wmark.height)[1]
            wm_value = 1 if tuptotal(wmark.getpixel((w_x, w_y))) > 0 else 0
            dither_value = d1[i][j] if wm_value == 1 else d0[i][j]
            orig_pix = img_orig.getpixel((i, j))
            cw_pixel = tdiff(quantum(tsum(orig_pix, dither_value), delta), dither_value)
            newimg.putpixel((i, j), tuple(tint(cw_pixel)))

    return newimg


def extract_wmark_dm_qim(key, delta, cw_img:Image.Image) -> Image.Image:
    d0, d1 = generate_dither_vectors(key, delta, cw_img.size)
    watermark = Image.new(cw_img.mode, cw_img.size)

    for i in range(cw_img.width):
        for j in range(cw_img.height):
            c_0 = tdiff(quantum(tsum(cw_img.getpixel((i, j)), d0[i][j]), delta), d0[i][j])
            c_1 = tdiff(quantum(tsum(cw_img.getpixel((i, j)), d1[i][j]), delta), d1[i][j])
            p_0 = tuptotal(tdiff(cw_img.getpixel((i, j)), c_0))
            p_1 = tuptotal(tdiff(cw_img.getpixel((i, j)), c_1))
            if p_0 <= p_1:
                watermark.putpixel((i, j), (0, 0, 0))
            else:
                watermark.putpixel((i, j), (255, 255, 255))

    return watermark


def embed_wmark_dct(img_orig: Image.Image, watermark: Image.Image) -> Image.Image:

    wm_expanded = expand_img(watermark, img_orig.size)
    wm_expanded.save('wm_expanded.bmp')
    newimg = fuse_two_images(img_orig, 0, 0, 1, wm_expanded, 0, 200, 0.2)
    return newimg


def tsum_dist(tuple_orig, tuple_app):
    res = []
    for i in range(len(tuple_orig)):
        res.append(tuple_orig[i] + tuple_app[i] / abs(tuple_app[i]-tuple_orig[i])
                   if tuple_app[i] != tuple_orig[i] else 1)
    return res


def extract_wmark_dct(size, img_wm: Image.Image) -> Image.Image:

    wid, hei = img_wm.width, img_wm.height

    watmark = Image.new(img_wm.mode, size)
    wm_prepared = empty_rgb_matrix(size[1], size[0])

    for i in range(wid):
        for j in range(hei):
            x = divmod(i, size[0])[1]
            y = divmod(j, size[1])[1]

            wm_prepared[x][y] = tsum_dist(wm_prepared[x][y], img_wm.getpixel((i,j)))
    for i in range(len(wm_prepared)):
        for j in range(len(wm_prepared[0])):
            wm_prepared[i][j] = tmul(wm_prepared[i][j], size[0] /wid * size[1]/ hei)
            watmark.putpixel((i,j), tuple(tint(wm_prepared[i][j])))
    return watmark





def gen_norm_random_mat(mat_size):

    output = np.zeros([*mat_size])
    for i in range(len(output)):
        for j in range(len(output[0])):
            output[i][j] = 2 * (random.random() - 0.5)
    return output


def gen_wm_blind_template(bit_size, mat_size):

    output = np.zeros([*mat_size], dtype=int)
    for i in range(len(output)):
        for j in range(len(output[0])):
            output[i][j] = int(bit_size * random.random())

    return output


def embed_wmark_blind_multi(alpha: float, bit_message:list, key, img_orig: Image.Image) -> Image.Image:
    wid, hei = img_orig.width, img_orig.height
    random.seed(key)
    wr = gen_norm_random_mat(img_orig.size)
    mat_helper = gen_wm_blind_template(len(bit_message), img_orig.size)
    newimg = img_orig.copy()
    for i in range(wid):
        for j in range(hei):
            wmod = wr[i][j] if bit_message[mat_helper[i][j]] == 1 else 0-wr[i][j]
            chan1 = img_orig.getpixel((i, j))[0]
            chan1 += alpha * wmod
            chan2 = img_orig.getpixel((i, j))[1]
            chan2 += alpha * wmod
            chan3 = img_orig.getpixel((i, j))[2]
            chan3 += alpha * wmod
            newimg.putpixel((i,j), tuple(tint([chan1, chan2, chan3])))

    return newimg


def p_blinder(p_k, threshold):
    if p_k > threshold:
        return 1
    elif p_k < 0-threshold:
        return 0
    else:
        return -1


def extract_wmark_blind_multi(thrsd, bit_len, key, img_cw: Image.Image):
    random.seed(key)
    wr = gen_norm_random_mat(img_cw.size)
    mat_helper = gen_wm_blind_template(bit_len, img_cw.size)

    b_restored = []
    for i in range(bit_len):
        print('extract k=', i)
        w_rk = np.zeros(wr.shape)
        num_k = 0
        for m in range(len(w_rk)):
            for n in range(len(w_rk[0])):
                if int(mat_helper[m][n]) == i:
                    w_rk[m][n] = wr[m][n]
                    num_k += 1
                else:
                    w_rk[m][n] = .0
        p_kred, p_kgreen, p_kblue = 0, 0, 0
        for m in range(img_cw.width):
            for n in range(img_cw.height):
                p_kred += img_cw.getpixel((m, n))[0] * w_rk[m][n]
                p_kgreen += img_cw.getpixel((m, n))[1] * w_rk[m][n]
                p_kblue += img_cw.getpixel((m, n))[2] * w_rk[m][n]
        cof = (1/num_k)
        p_kred *= cof
        p_kgreen *= cof
        p_kblue *= cof
        b_kr, b_kg, b_kb = p_blinder(p_kred, thrsd), p_blinder(p_kgreen, thrsd), p_blinder(p_kblue, thrsd)
        b_restored.append((b_kr, b_kg, b_kb))

    return b_restored

image_name = 'car-jeep'
wm_name = 'wm'
folder = 'examples/'
one: Image.Image = Image.open(folder + image_name + '.bmp')

key = 'hello'
bit_mes = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

emb_img = embed_wmark_blind_multi(8, bit_mes, key, one)
emb_img.save(image_name+'_with_blind_multi_wm('+key+').bmp')
print('embeded')
b_res = extract_wmark_blind_multi(0.2, len(bit_mes), key, emb_img)
print('extracted')
print(b_res)

# image_name = 'car-fast'
# wm_name = 'wm'
# folder = 'examples/'
# # # second_image_name = 'sun-and-sky'
# # #
# one: Image.Image = Image.open(folder + image_name + '.bmp')
# two: Image.Image = Image.open(wm_name+'.bmp')
# key = 'hello'
# delta = 16
# print('embeding...')
# three = embed_wmark_dct(one, two)
# # three = embed_wmark_dm_qim(key, delta, one, two)
# three.save(image_name+'_with_dct_'+wm_name+'.bmp')
# print('extracting...')
# extracted = extract_wmark_dct(two.size, three)
# # extracted = extract_wmark_dm_qim(key, delta, three)
# extracted.save(image_name+'dct_wm_extracted.bmp')


#
# newimage = sum_two_images(one, 0.4, two, 0.6)
#
# newimage.save(image_name + '_and_' + second_image_name + '_as_new_img.bmp')
# noise_set = get_noisy_set(0.9, 50, orig)
# print('GOT SET')
# sample: Image.Image = noise_set[0]
# sample.save('impulse/'+image_name+'_noise_sample.bmp')
# clear_image = impulse_fusion_filter(noise_set)
# print('FINISHED')
# clear_image.save('impulse/'+image_name + '_clear.bmp')

# orig = Image.open(image_name + '.bmp')
#
# t_frames = scan(8, orig)
# print('matsynth started')
# new_image = mat_synth(t_frames)
# new_image.save(image_name+'_matrix.bmp')
