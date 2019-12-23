import PIL.Image as Image
from additions import tint, tup_sum, tup_dif, tup_total, tup_mul, empty_rgb_matrix, pave_image
from fusions import fusion_two_images


def embed_wmark_additive(alpha: float, img_orig: Image.Image, watermark: Image.Image) -> Image.Image:
    newimg = img_orig.copy()
    wm_expanded = pave_image(watermark, img_orig.size)
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
            newimg.putpixel((i, j), tup_sum(orig_pixel, tup_mul(wm_expanded.getpixel((i, j)), alpha * pelmeni)))

    return newimg


def quantum(channels, delta):
    output = []
    for el in channels:
        output.append(delta * int((el + 0.5) / delta))
    return output


def generate_dither_vectors(key, delta, size):
    import random
    import numpy as np
    random.seed(key)
    d0 = np.zeros([*size, 3])
    d1 = np.zeros([*size, 3])
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(3):
                r = random.random()
                d0[i][j][k] = int(delta * (r - 0.5))
                d1[i][j][k] = int(d0[i][j][k] - np.sign(d0[i][j][k]) * (delta / 2))

    return d0, d1


def embed_wmark_dm_qim(key, delta, img_orig: Image.Image, wmark: Image.Image) -> Image.Image:
    newimg = img_orig.copy()
    d0, d1 = generate_dither_vectors(key, delta, img_orig.size)
    for i in range(img_orig.width):
        for j in range(img_orig.height):
            w_x, w_y = divmod(i, wmark.width)[1], divmod(j, wmark.height)[1]
            wm_value = 1 if tup_total(wmark.getpixel((w_x, w_y))) > 0 else 0
            dither_value = d1[i][j] if wm_value == 1 else d0[i][j]
            orig_pix = img_orig.getpixel((i, j))
            cw_pixel = tup_dif(quantum(tup_sum(orig_pix, dither_value), delta), dither_value)
            newimg.putpixel((i, j), tuple(tint(cw_pixel)))

    return newimg


def extract_wmark_dm_qim(key, delta, cw_img: Image.Image) -> Image.Image:
    d0, d1 = generate_dither_vectors(key, delta, cw_img.size)
    watermark = Image.new(cw_img.mode, cw_img.size)

    for i in range(cw_img.width):
        for j in range(cw_img.height):
            c_0 = tup_dif(quantum(tup_sum(cw_img.getpixel((i, j)), d0[i][j]), delta), d0[i][j])
            c_1 = tup_dif(quantum(tup_sum(cw_img.getpixel((i, j)), d1[i][j]), delta), d1[i][j])
            p_0 = tup_total(tup_dif(cw_img.getpixel((i, j)), c_0))
            p_1 = tup_total(tup_dif(cw_img.getpixel((i, j)), c_1))
            if p_0 <= p_1:
                watermark.putpixel((i, j), (0, 0, 0))
            else:
                watermark.putpixel((i, j), (255, 255, 255))

    return watermark


def embed_wmark_dct(img_orig: Image.Image, watermark: Image.Image) -> Image.Image:
    wm_expanded = pave_image(watermark, img_orig.size)
    wm_expanded.save('wm_expanded.bmp')
    newimg = fusion_two_images(img_orig, 0, 0, 1, wm_expanded, 0, 200, 0.2)
    return newimg


def tsum_dist(tuple_orig, tuple_app):
    res = []
    for i in range(len(tuple_orig)):
        res.append(tuple_orig[i] + tuple_app[i] / abs(tuple_app[i] - tuple_orig[i])
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

            wm_prepared[x][y] = tsum_dist(wm_prepared[x][y], img_wm.getpixel((i, j)))
    for i in range(len(wm_prepared)):
        for j in range(len(wm_prepared[0])):
            wm_prepared[i][j] = tup_mul(wm_prepared[i][j], size[0] / wid * size[1] / hei)
            watmark.putpixel((i, j), tuple(tint(wm_prepared[i][j])))
    return watmark


def gen_norm_random_mat(mat_size):
    import random
    import numpy as np
    output = np.zeros([*mat_size])
    for i in range(len(output)):
        for j in range(len(output[0])):
            output[i][j] = 2 * (random.random() - 0.5)
    return output


def gen_wm_blind_template(bit_size, mat_size):
    import random
    import numpy as np
    output = np.zeros([*mat_size], dtype=int)
    for i in range(len(output)):
        for j in range(len(output[0])):
            output[i][j] = int(bit_size * random.random())

    return output


def embed_wmark_blind_multi(gain_alpha: float, bit_message: list, key, img: Image.Image) -> Image.Image:
    import random
    img_new = img.copy()
    random.seed(key)
    bit_len = len(bit_message)
    for i in range(img.width):
        for j in range(img.height):
            wr_ij = 2 * (random.random() - 0.5)
            w_mod = wr_ij if bit_message[int(bit_len * random.random())] == 1 else -wr_ij
            g_times_wm = gain_alpha * w_mod
            old_pix = img.getpixel((i, j))
            new_pix = int(old_pix[0] * g_times_wm), int(old_pix[1] * g_times_wm), int(old_pix[2] * g_times_wm)
            img_new.putpixel((i, j), new_pix)

    return img_new


def p_blinder(p_k, threshold):
    res = []
    for elm in p_k:
        if elm > threshold:
            res.append(1)
        elif elm < 0 - threshold:
            res.append(0)
        else:
            res.append(-1)
    return res


def extract_wmark_blind_multi(thrsd, bit_len, key, img: Image.Image):
    import random

    b_restored = []
    for b_k in range(bit_len):
        print('extraction of', b_k + 1, ' bit')
        num_k = 0
        p_k_tuple = [0, 0, 0]
        random.seed(key)
        for i in range(img.width):
            for j in range(img.height):
                wr_ij = 2 * (random.random() - 0.5)
                mat_h_ij = int(bit_len * random.random())

                pix = list(img.getpixel((i, j)))
                if mat_h_ij == b_k:
                    w_rk_val = wr_ij
                    num_k += 1
                else:
                    w_rk_val = .0

                p_k_tuple[0] += pix[0] * w_rk_val
                p_k_tuple[1] += pix[1] * w_rk_val
                p_k_tuple[2] += pix[2] * w_rk_val

        cof = 1 / num_k
        p_k_tuple[0] *= cof
        p_k_tuple[1] *= cof
        p_k_tuple[2] *= cof

        b_restored.append(p_blinder(p_k_tuple, thrsd))

    b_out = []
    for elm in b_restored:
        elm.sort()
        b_out.append(elm[1])
    return b_out
