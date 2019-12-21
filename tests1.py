import os
import PIL.Image as Image
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt
from skimage.io import imread, imsave, imshow
from math import sqrt


def get_empty_matrix(lng, wid) -> list:
    return [[[0, 0, 0] for i in range(lng)] for j in range(wid)]


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


def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def image_expand(img, m_w, m_h):
    import numpy as np
    img_w = img.shape[0]
    img_h = img.shape[1]
    cast_img = np.array([[0] * (img_h + m_h - 1)] * (img_w + m_w - 1)).astype(np.uint8)

    cast_img[:1 - m_w, :m_h] = img[:, :m_h]
    cast_img[-m_w:, :1 - m_h] = img[-m_w:, :]
    cast_img[m_w - 1:, -m_h:] = img[:, -m_h:]
    cast_img[:m_w, m_h - 1:] = img[:m_w, :]

    cast_img[m_w // 2:-m_w + 2, m_h // 2:-m_h + 2] = img[:, :]
    return cast_img


def mat_synth(frames: list) -> Image.Image:
    shot_h = len(frames[0])
    shot_w = len(frames)
    low_width = frames[0][0].width
    low_height = frames[0][0].height
    top_width = shot_w * low_width
    top_height = shot_h * low_height
    img = get_empty_matrix(top_width, top_height)

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
            out.putpixel((i,j), tuple(img[i][j]))
    return out


def scan(scale, orig_img: Image.Image) -> list:
    print('size of original image [', orig_img.width, orig_img.height, ']')
    mask_w = orig_img.width - scale
    mask_h = orig_img.height - scale
    new_size_w = int(orig_img.width / scale)
    new_size_h = int(orig_img.height / scale)
    print('Then mask size is [', mask_w, mask_h, ']')
    print('Output image size is [', new_size_w, new_size_h, ']')
    if not os.path.exists('results'):
        os.mkdir('results')

    frames = []

    for i in range(scale):
        j_frames = []
        for j in range(scale):
            new_img = orig_img.crop((i, j, mask_w + i, mask_h + j))
            new_img = new_img.resize((new_size_w, new_size_h), Image.NEAREST)
            j_frames = j_frames + [new_img]
        frames = frames + [j_frames]

    xs = 0
    for j_frames in frames:
        xs = xs + 1
        ys = 0
        for frame in j_frames:
            ys = ys + 1
            frame.resize((orig_img.width, orig_img.height), Image.LINEAR).save('results/scan_yAxis' + str(ys) + '_xAxis' + str(xs) + '.bmp')
    return frames


def scan2(scale, orig_img: Image.Image) -> list:

    print('size of original image [', orig_img.width, orig_img.height, ']')
    mask_size = scale
    new_size_w = int(orig_img.width / scale)
    new_size_h = int(orig_img.height / scale)
    print('Then mask size is [', mask_size, mask_size, ']')
    print('Output image size is [', new_size_w, new_size_h, ']')
    if not os.path.exists('results'):
        os.mkdir('results')
    frames = []
    mask = get_empty_matrix(mask_size, mask_size)
    mask_half = mask_size // 2
    for x in range(scale):
        j_frames = []
        for y in range(scale):
            tmp_img = Image.new('RGB', (new_size_w, new_size_h))
            for i in range(new_size_w):
                isc = i * scale + x
                for j in range(new_size_h):
                    jsc = j*scale+y
                    for m1 in range(0-mask_half+1, mask_size-mask_half+1, 1):
                        for m2 in range(0-mask_half+1, mask_size-mask_half+1, 1):
                            isc_mx = isc+m1
                            jsc_my = jsc+m2
                            if 0 <= isc_mx < orig_img.width and 0 <= jsc_my < orig_img.height:
                                mask[m1][m2] = orig_img.getpixel((isc_mx, jsc_my))
                            else:
                                mask[m1][m2] = (0, 0, 0)

                    mean = tuple(mean_of_mat(mask))

                    tmp_img.putpixel((i, j), mean)
            j_frames = j_frames + [tmp_img]
            tmp_img.resize((orig_img.width, orig_img.height), Image.LINEAR).save('results/scan_yAxis' + str(y) + '_xAxis' + str(x) + '.bmp')
        frames = frames + [j_frames]

    return frames


orig = Image.open('sunflowers.bmp')
t_frames = scan(32, orig)
new_image = mat_synth(t_frames)
new_image.save('sunflowers_new.bmp')
