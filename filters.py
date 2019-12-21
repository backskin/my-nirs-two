def hist(img):
    histogram = [0] * 256
    for i in range(len(img)):
        for j in range(len(img[i])):
            histogram[img[i, j]] += 1
    return histogram


def lin_con_uint(img, precision):
    import numpy as np
    x_min, x_max = np.percentile(img, [100*precision, 100*(1 - precision)])
    img_out = 255 * (img - x_min) / (x_max - x_min)
    return np.clip(img_out, 0, 255).astype(np.uint8)


def lin_con(img, precision):
    import numpy as np
    x_min, x_max = np.percentile(img, [100*precision, 100*(1 - precision)])
    img_out = (img - x_min) / (x_max - x_min)
    return np.clip(img_out, 0.0, 1.0)


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


def convolution(img, mask):
    import numpy as np
    m_w = len(mask)
    m_h = len(mask[0])

    new_img = img.copy()
    cast_img = image_expand(img, m_w, m_h)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cast_block = cast_img[i: i + m_w, j: j + m_h]
            new_img[i, j] = np.clip((cast_block * mask[:, :]).sum(), 0, 255)

    return new_img


def convolution_rgb(img, mask):
    img[:, :, 0] = convolution(img[:, :, 0], mask)
    img[:, :, 1] = convolution(img[:, :, 1], mask)
    img[:, :, 2] = convolution(img[:, :, 2], mask)

    return img


def borders(img, n):
    import numpy as np

    new_img = img.copy()
    cast_img = image_expand(img, n, n)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cast_block = cast_img[i: i + n, j: j + n]
            over_d = 0
            under_d = 0
            for m in range(n-1):
                over_d += np.diag(cast_block, m+1).sum()
                under_d += np.diag(cast_block, -m-1).sum()
            if over_d >= under_d:
                new_img[i, j] = 1
            else:
                new_img[i, j] = 0
    return new_img



def convolution_yuv(img, mask):
    from bt709 import to_yuv, to_rgb
    from skimage import img_as_float, img_as_ubyte
    import numpy as np

    cont_yuv = to_yuv(img_as_float(img))
    cont_yuv[:, :, 0] = img_as_float(convolution(img_as_ubyte(cont_yuv[:, :, 0]), mask))
    cont_rgb = img_as_ubyte(np.clip(to_rgb(cont_yuv), -1, 1))

    return cont_rgb


def variative_arr(block, mask):
    import numpy as np
    if len(block) != len(mask):
        raise Exception('Exception during filtering:', 'block.len %d != mask.len %d' % (len(block), len(mask)))

    block = np.array(block).flatten()
    mask = np.array(mask).flatten()
    if len(mask) % 2 == 0:
        raise Exception('Exception during filtering:' 'mask.len is even; must be odd number')

    array = []
    for i in range(len(mask)):
        for j in range(mask[i]):
            array += [block[i]]

    array.sort()
    return array


def dilatation(block, mask):
    return variative_arr(block, mask)[-1]


def erosion(block, mask):
    return variative_arr(block, mask)[0]


def median(block, mask):
    v_arr = variative_arr(block, mask)
    return v_arr[len(v_arr) // 2 + 1]


def rng_filter(method, img, mask):
    import numpy as np
    m_w = len(mask)
    m_h = len(mask[0])
    new_img = img.copy()
    cast_img = image_expand(img, m_w, m_h)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cast_block = np.array(cast_img[i: i + m_w, j: j + m_h])
            new_img[i, j] = method(cast_block, mask)

    return new_img


def noising(img, k):
    import numpy as np
    noise_mask = np.array(
        [[int(np.random.normal(0, k, 1)[0])
          for j in range(img.shape[1])] for i in range(img.shape[0])])
    img[:, :] = np.clip(img[:, :] + noise_mask[:, :], 0, 255)
    return img


def noising_rgb(img, ratio=1):
    img[:, :, 0] = noising(img[:, :, 0], ratio)
    img[:, :, 1] = noising(img[:, :, 1], ratio)
    img[:, :, 2] = noising(img[:, :, 2], ratio)

    return img


def noising_yuv(img, ratio=1):
    from bt709 import to_yuv, to_rgb
    from skimage import img_as_float, img_as_ubyte
    import numpy as np

    cont_yuv = to_yuv(img_as_float(img))
    cont_yuv[:, :, 0] = img_as_float(noising(img_as_ubyte(cont_yuv[:, :, 0]), ratio))
    cont_rgb = img_as_ubyte(np.clip(to_rgb(cont_yuv), -1, 1))

    return cont_rgb


def rng_filter_rgb(method, img, mask=None):
    import numpy as np
    if mask is None:
        mask = np.array([[1] * 3] * 3)

    img[:, :, 0] = rng_filter(method, img[:, :, 0], mask)
    img[:, :, 1] = rng_filter(method, img[:, :, 1], mask)
    img[:, :, 2] = rng_filter(method, img[:, :, 2], mask)

    return img


def rng_filter_yuv(method, img, mask=None):
    from bt709 import to_yuv, to_rgb
    from skimage import img_as_float, img_as_ubyte
    import numpy as np
    if mask is None:
        mask = np.array([[1] * 3] * 3)

    cont_yuv = to_yuv(img_as_float(img))
    cont_yuv[:, :, 0] = img_as_float(rng_filter(method, img_as_ubyte(cont_yuv[:, :, 0]), mask))
    cont_rgb = img_as_ubyte(np.clip(to_rgb(cont_yuv), -1, 1))

    return cont_rgb


def similarity(img1, img2):
    # TO-DO: поставить сравнение размеров
    import numpy as np
    img1 = np.array(img1).astype(np.int)
    img2 = np.array(img2).astype(np.int)
    # если вычесть белый шум, то получится в среднем половина попаданий
    # поэтому нужно результат сместить на 0.5 и умножить на 2
    img_sub = np.absolute(np.clip(img1 - img2, -1, 1)).sum()
    sqr = img1.shape[0] * img1.shape[1]
    return np.clip(2 * (sqr//2 - img_sub) / sqr, 0, 1)
