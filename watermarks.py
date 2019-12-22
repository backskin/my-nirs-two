import PIL.Image as Image
from additions import tint, tsum, tdiff, tup_total, tmul, empty_rgb_matrix
from fusions import fusion_two_images


def expand_img(img:Image.Image, new_size) -> Image.Image:
    newimg = Image.new(img.mode, new_size)
    for i in range(newimg.width):
        for j in range(newimg.height):
            newimg.putpixel((i,j), img.getpixel((divmod(i, img.width)[1], divmod(j, img.height)[1])))
    return newimg


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
    import random
    import numpy as np
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
            wm_value = 1 if tup_total(wmark.getpixel((w_x, w_y))) > 0 else 0
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
            p_0 = tup_total(tdiff(cw_img.getpixel((i, j)), c_0))
            p_1 = tup_total(tdiff(cw_img.getpixel((i, j)), c_1))
            if p_0 <= p_1:
                watermark.putpixel((i, j), (0, 0, 0))
            else:
                watermark.putpixel((i, j), (255, 255, 255))

    return watermark


def embed_wmark_dct(img_orig: Image.Image, watermark: Image.Image) -> Image.Image:

    wm_expanded = expand_img(watermark, img_orig.size)
    wm_expanded.save('wm_expanded.bmp')
    newimg = fusion_two_images(img_orig, 0, 0, 1, wm_expanded, 0, 200, 0.2)
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


def embed_wmark_blind_multi(alpha: float, bit_message:list, key, img_orig: Image.Image) -> Image.Image:
    import random
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
    import random
    import numpy as np
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