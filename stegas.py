def nzb_insert(contan, watermark):
    new_cont = contan.copy()
    st_i = len(watermark)
    st_j = len(watermark[0])

    for i in range(len(contan) // st_i):
        for j in range(len(contan[0]) // st_j):
            old_bit = new_cont[i * st_i:(i + 1) * st_i, j * st_j:(j + 1) * st_j] % 2
            new_cont[i * st_i:(i + 1) * st_i, j * st_j:(j + 1) * st_j] -= \
                old_bit[:, :] - (watermark[:, :] % 2 + (j + i) % 2) % 2

    return new_cont


def nzb_extract(container):
    plot = container.copy()
    plot[:, :] = (container[:, :] % 2) * 255
    return plot


def nzb_insert_secured(contan, mark, key):
    import random
    import numpy as np

    new_cont = contan.copy()
    st_i = len(mark)
    st_j = len(mark[0])

    random.seed(key)
    key_mask = np.array([[random.getrandbits(1)
                          for j in range(new_cont.shape[1])]
                         for i in range(new_cont.shape[0])]).astype(np.uint8)

    for i in range(len(contan) // st_i):
        for j in range(len(contan[0]) // st_j):
            old_bit = new_cont[i * st_i:(i + 1) * st_i, j * st_j:(j + 1) * st_j] % 2
            new_cont[i * st_i:(i + 1) * st_i, j * st_j:(j + 1) * st_j] -= \
                old_bit[:, :] - (key_mask[i * st_i:(i + 1) * st_i, j * st_j:(j + 1) * st_j]
                                 + mark[:, :] + j + i) % 2

    return new_cont


def nzb_extract_secured(container, key):
    import random
    random.seed(key)
    plot = container.copy()
    for i in range(len(container)):
        for j in range(len(container[0])):
            sec_bit = random.getrandbits(1)
            plot[i, j] = ((container[i, j] + sec_bit) % 2) * 255

    return plot


def insert_dwm(rgb_container, dwm):
    from bt709 import to_yuv, to_rgb
    from skimage import img_as_float, img_as_ubyte
    import numpy as np

    cont_yuv = to_yuv(img_as_float(rgb_container))
    cont_yuv[:, :, 0] = img_as_float(nzb_insert(img_as_ubyte(cont_yuv[:, :, 0]), dwm))
    cont_rgb = img_as_ubyte(np.clip(to_rgb(cont_yuv), -1, 1))

    return cont_rgb


def extract_dwm(rgb_container):
    from bt709 import to_yuv
    from skimage import img_as_float, img_as_ubyte
    import numpy as np
    cont_yuv = to_yuv(img_as_float(rgb_container))
    dwm_e = nzb_extract(img_as_ubyte(np.clip(cont_yuv[:, :, 0], -1, 1)))

    return dwm_e


def insert_dwm_wkey(rgb_container, dwm, key):
    from bt709 import to_yuv, to_rgb
    from skimage import img_as_float, img_as_ubyte
    import numpy as np

    cont_yuv = to_yuv(img_as_float(rgb_container))
    cont_yuv[:, :, 0] = img_as_float(nzb_insert_secured(img_as_ubyte(cont_yuv[:, :, 0]), dwm, key))
    cont_rgb = img_as_ubyte(np.clip(to_rgb(cont_yuv), -1, 1))

    return cont_rgb


def extract_dwm_wkey(rgb_container, key):
    from bt709 import to_yuv
    from skimage import img_as_float, img_as_ubyte

    cont_yuv = to_yuv(img_as_float(rgb_container))
    dwm_e = nzb_extract_secured(img_as_ubyte(cont_yuv[:, :, 0]), key)

    return dwm_e


def dwm_guess(dwm, orig_w, orig_h):
    import numpy as np
    mid_dwm = np.array([[0] * orig_h] * orig_w).astype(np.uint8)
    mid_dwm[:, :] = mid_dwm[:, :] * 0
    for i in range(len(dwm) // orig_w):
        for j in range(len(dwm[0]) // orig_h):
            mid_dwm[:, :] \
                = np.clip(mid_dwm[:, :] + pow(-1, i + j)
                          * (2 * (-dwm[i * orig_w:(i + 1) * orig_w, j * orig_h:(j + 1) * orig_h] % 2) + 1), 0, 255)

    threshold = len(dwm) // orig_w * len(dwm[0]) // orig_h // 8

    for i in range(len(mid_dwm)):
        for j in range(len(mid_dwm[0])):
            mid_dwm[i, j] = 255 if mid_dwm[i, j] > threshold else 0

    return mid_dwm