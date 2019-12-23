import PIL.Image as Image
from additions import empty_rgb_matrix, expand_image, tup_sum, tup_mul, tint, adjust_channels, \
    high_pass_filter, low_pass_filter, dct2, idct2, tup_total, tup_sort, tup_median
import numpy as np


def fusion_superresolution(frames: list) -> Image.Image:
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


def scan_superresolution(scale, orig_img: Image.Image) -> list:
    import os
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
            j_frames = j_frames + [new_img]
        frames = frames + [j_frames]

    return frames


def scan_blurry(scale, orig_img: Image.Image) -> list:
    import os
    cast_img = expand_image(orig_img, scale + 1, scale + 1)
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
                                mean = tup_sum(mean, cast_img.getpixel((isc + m1, jsc + m2)))
                    mean = tup_mul(mean, 1 / (scale * scale))
                    tmp_img.putpixel((i, j), tuple(tint(mean)))

            j_frames = j_frames + [tmp_img]
        frames = frames + [j_frames]

    return frames


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


def sum_cha_packs(pack_one, pack_two):
    return pack_one[0] + pack_two[0], pack_one[1] + pack_two[1], pack_one[2] + pack_two[2]


def get_image_from_channels(ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray) -> Image.Image:
    size = (len(ch1), len(ch1[0]))
    newimg = Image.new('RGB', size)
    for i in range(size[0]):
        for j in range(size[1]):
            pixel = [ch1[i][j], ch2[i][j], ch3[i][j]]
            newimg.putpixel((i, j), tuple(tint(pixel)))
    return newimg


def fusion_two_images(img1: Image.Image, b1: int, c1: int, k1: float,
                      img2: Image.Image, b2: int, c2: int, k2: float) -> Image.Image:
    cha_pack1 = adjust_channels(k1, *toFreq(img1))
    if b1 > 0:
        cha_pack1 = low_pass_filter(b1, *cha_pack1)
    if c1 > 0:
        cha_pack1 = high_pass_filter(c1, *cha_pack1)
    cha_pack2 = adjust_channels(k2, *toFreq(img2))

    if b2 > 0:
        cha_pack2 = low_pass_filter(b2, *cha_pack2)
    if c2 > 0:
        cha_pack2 = high_pass_filter(c2, *cha_pack2)

    return get_image_from_channels(*fromFreq(*sum_cha_packs(cha_pack1, cha_pack2)))


def get_bitmaps(noi_list: list) -> list:
    result = []
    image: Image.Image
    for image in noi_list:
        bitmap = [[0 if 0 < tup_total(image.getpixel((i, j))) < 765 else 1
                   for i in range(image.height)]
                  for j in range(image.width)]
        result += [bitmap]
    return result


def get_smallest_bitmap(bitmaps: list):
    smallest = 0
    sum_val = np.array(bitmaps[smallest]).sum()
    for i in range(len(bitmaps)):
        t_val = np.array(bitmaps[i]).sum()
        if t_val < sum_val:
            smallest = i
            sum_val = t_val
    return smallest, bitmaps[smallest]


def fusion_impulse_filter(img_list: list) -> Image.Image:
    while len(img_list) > 1:
        num, bm = get_smallest_bitmap(get_bitmaps(img_list))
        print('Impulse fusion: ESTIMATED ', len(img_list))
        best_sample = img_list.pop(num)
        for image in img_list:
            for i in range(image.width):
                for j in range(image.height):
                    if bm[i][j] == 0:
                        image.putpixel((i, j), best_sample.getpixel((i, j)))

    return img_list[0]


def fusion_stacking(fusion_type: str, shots: list) -> Image.Image:

    example: Image.Image = shots[0]
    if fusion_type != 'mean' and fusion_type != 'median':
        print('STACKING: ERROR. There is no ', fusion_type, ' type')
        return example
    img_new = Image.new(example.mode, example.size)
    print('stacking...')
    if fusion_type == 'mean':
        for i in range(example.width):
            for j in range(example.height):
                pix = [0, 0, 0]
                for shot in shots:
                    shot_pix = shot.getpixel((i, j))
                    pix[0] += shot_pix[0]
                    pix[1] += shot_pix[1]
                    pix[2] += shot_pix[2]
                img_new.putpixel((i, j), tuple(tint(tup_mul(pix, (1 / len(shots))))))

    else:
        shot: Image.Image
        for i in range(example.width):
            for j in range(example.height):
                pixes = []
                for shot in shots:
                    pixes.append(shot.getpixel((i, j)))
                img_new.putpixel((i, j), tuple(tint(tup_median(tup_sort(pixes)))))

    return img_new
