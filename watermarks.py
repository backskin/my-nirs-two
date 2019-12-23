import PIL.Image as Image
from additions import tint, tup_sum, tup_dif, tup_total, tup_mul, empty_rgb_matrix, pave_image
from fusions import fusion_two_images


def embed_wmark_additive(alpha: float, img_orig: Image.Image, watermark: Image.Image) -> Image.Image:
    img_new = img_orig.copy()

    for i in range(img_new.width):
        for j in range(img_new.height):
            pix = img_orig.getpixel((i, j))
            wm_pix = watermark.getpixel((i % watermark.width, j % watermark.height))
            if wm_pix[0] == 0:
                continue
            img_new.putpixel((i, j), (pix[0] + int(wm_pix[0] * alpha * (-1 if pix[0] > 128 else 1)),
                                      pix[1] + int(wm_pix[1] * alpha * (-1 if pix[1] > 128 else 1)),
                                      pix[2] + int(wm_pix[2] * alpha * (-1 if pix[2] > 128 else 1))))

    return img_new


def quantum(channels, delta):
    output = []
    for el in channels:
        output.append(delta * int((el + 0.5) / delta))
    return output


# def generate_dither_vectors(key, delta, size):
#     import random
#     import numpy as np
#     random.seed(key)
#     d0 = np.zeros([*size, 3])
#     d1 = np.zeros([*size, 3])
#     for i in range(size[0]):
#         for j in range(size[1]):
#             for k in range(3):
#                 r = random.random()
#                 d0[i][j][k] = int(delta * (r - 0.5))
#                 d1[i][j][k] = int(d0[i][j][k] - np.sign(d0[i][j][k]) * (delta / 2))
#
#     return d0, d1


def embed_wm_dm_qim(key, delta, orig_img: Image.Image, watermark: Image.Image) -> Image.Image:
    import random
    from numpy import sign
    new_img = orig_img.copy()
    wm_wid = watermark.width
    wm_hei = watermark.height
    # d0, d1 = generate_dither_vectors(key, delta, orig_img.size)
    dlh = delta / 2
    random.seed(key)
    for i in range(orig_img.width):
        for j in range(orig_img.height):
            elem = int(delta * (random.random() - 0.5))
            sec_elem = int(elem - sign(elem) * dlh)
            d0_ij = elem, elem, elem
            d1_ij = sec_elem, sec_elem, sec_elem

            w_pixel = watermark.getpixel((i % wm_wid, j % wm_hei))
            dither_value = d1_ij if tup_total(w_pixel) > 0 else d0_ij
            orig_pix = list(orig_img.getpixel((i, j)))
            cw_pixel = tup_dif(quantum(tup_sum(orig_pix, dither_value), delta), dither_value)
            new_img.putpixel((i, j), tuple(tint(cw_pixel)))
    return new_img


def extract_wm_dm_qim(key, delta, cw_img: Image.Image) -> Image.Image:
    import random
    from numpy import sign
    # d0, d1 = generate_dither_vectors(key, delta, cw_img.size)
    watermark = Image.new(cw_img.mode, cw_img.size)
    random.seed(key)
    dlh = delta / 2
    for i in range(cw_img.width):
        for j in range(cw_img.height):
            pix = list(cw_img.getpixel((i, j)))

            elem = int(delta * (random.random() - 0.5))
            sec_elem = int(elem - sign(elem) * dlh)
            d0_ij = elem, elem, elem
            d1_ij = sec_elem, sec_elem, sec_elem

            c_0 = tup_dif(quantum(tup_sum(pix, d0_ij), delta), d0_ij)
            c_1 = tup_dif(quantum(tup_sum(pix, d1_ij), delta), d1_ij)
            p_0 = tup_total(tup_dif(pix, c_0))
            p_1 = tup_total(tup_dif(pix, c_1))
            if p_0 <= p_1:
                watermark.putpixel((i, j), (0, 0, 0))
            else:
                watermark.putpixel((i, j), (255, 255, 255))

    return watermark


def qim_extract_result(wm_extracted, wm_orig) -> float:
    return 0


def embed_wm_dct(img_orig: Image.Image, watermark: Image.Image) -> Image.Image:
    wm_expanded = pave_image(watermark, img_orig.size)
    wm_expanded.save('wm_expanded.bmp')
    newimg = fusion_two_images(img_orig, 0, 0, 1, wm_expanded, 0, 50, 0.2)
    return newimg


def tsum_dist(tuple_orig, tuple_app):
    res = []
    for i in range(len(tuple_orig)):
        res.append(tuple_orig[i] + tuple_app[i] / abs(tuple_app[i] - tuple_orig[i])
                   if tuple_app[i] != tuple_orig[i] else 1)
    return res


def extract_wm_dct(size, img_wm: Image.Image) -> Image.Image:
    wid, hei = img_wm.width, img_wm.height

    watermark = Image.new(img_wm.mode, size)
    wm_prepared = empty_rgb_matrix(size[1], size[0])

    for i in range(wid):
        for j in range(hei):
            x = i % size[0]
            y = j % size[1]

            wm_prepared[x][y] = tsum_dist(wm_prepared[x][y], img_wm.getpixel((i, j)))
    for i in range(len(wm_prepared)):
        for j in range(len(wm_prepared[0])):
            wm_prepared[i][j] = tup_mul(wm_prepared[i][j], size[0] / wid * size[1] / hei)
            watermark.putpixel((i, j), tuple(tint(wm_prepared[i][j])))
    return watermark


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


def embed_wm_blind_multi(gain_alpha: float, bit_message: list, key, img: Image.Image) -> Image.Image:
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
            new_pix = old_pix[0] + int(old_pix[0] * g_times_wm), \
                      old_pix[1] + int(old_pix[1] * g_times_wm), \
                      old_pix[2] + int(old_pix[2] * g_times_wm)

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


def blind_extract_result(b_array: list, b_restored: list) -> float:
    result = 0
    for i in range(len(b_array)):
        if b_restored[i] == b_array[i]:
            result += 1
    return result / len(b_array)
