from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from filters import rng_filter_yuv, median, erosion, dilatation, \
    convolution_rgb, convolution_yuv, noising_yuv, similarity
from stegas import insert_dwm, extract_dwm, insert_dwm_wkey, \
    extract_dwm_wkey, dwm_guess, nzb_insert, nzb_extract

# # TEST 1 : simple inception
# #
# image, dwm = imread('palm-tree.png'), imread('dwm3.bmp')
# im_with_dwm = insert_dwm(image, dwm)
# imsave('cont-with-dwm.png', im_with_dwm)
# again_img = imread('cont-with-dwm.png')
# dwm_layer = extract_dwm(again_img)
# imsave('dwm-layer-1.png', dwm_layer)
# imshow(dwm_layer)
# plt.show()
#
# img, dwm = imread('bw-flower.png'), imread('dwm2.bmp')
# n_i = nzb_insert(img, dwm)
# imsave('bw-flower-nzb.png', n_i)
# imsave('bw-fl-layer.png', nzb_extract(n_i))
#
#
# # TEST 2: inception with key
# #
image, dwm = imread('lenna.png'), imread('dwm3.bmp')

key = 'hello_world'
wrong_key = 'bye_bye'

im_with_dwm = insert_dwm_wkey(image, dwm, key)
imsave('cont-with-dwm.png', im_with_dwm)
again_img = imread('cont-with-dwm.png')
wr_dwm_layer = extract_dwm_wkey(again_img, wrong_key)
imshow(wr_dwm_layer)
plt.show()
dwm_layer = extract_dwm_wkey(again_img, key)
imshow(dwm_layer)
plt.show()
#
# # TEST 3: check simple inception and extraction after noise
# #
# image, dwm = imread('red-flower.png'), imread('dwm2.bmp')
# im_with_dwm = insert_dwm(image, dwm)
# imsave('cont-with-dwm.png', im_with_dwm)
# imshow(im_with_dwm)
# plt.show()
# dwm_layer = extract_dwm(im_with_dwm)
# imshow(dwm_layer)
# plt.show()
# im_with_dwm = noising_yuv(im_with_dwm.copy(), 2)
# imsave('cont-with-dwm-noised.png', im_with_dwm)
# again_img = imread('cont-with-dwm-noised.png')
# imshow(again_img)
# plt.show()
# dwm_layer = extract_dwm(again_img)
# imshow(dwm_layer)
# plt.show()
# dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
# imsave('dwm-restored-from-noise(2).png', dwm_g)
# imshow(dwm_g)
# plt.show()
#
# # TEST 4: check simple inc and ext after sharpen (all three RGB channels)
# #
# image, dwm = imread('tiger-color.png'), imread('dwm2.bmp')
# sharp_mask = 1 / 8 * np.array([[-1, -1, -1],
#                                [-1, 16, -1],
#                                [-1, -1, -1]])[:, :]
#
# im_with_dwm = insert_dwm(image, dwm)
# dwm_layer = extract_dwm(im_with_dwm)
# imshow(dwm_layer)
# plt.show()
# im_with_dwm = convolution_rgb(im_with_dwm.copy(), sharp_mask)
# imsave('cont-with-dwm-sharp.png', im_with_dwm)
# again_img = imread('cont-with-dwm-sharp.png')
# imshow(again_img)
# plt.show()
# dwm_layer = extract_dwm(again_img)
# imsave('dwm-layer-sharpen.png', dwm_layer)
# imshow(dwm_layer)
# plt.show()
# dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
# imsave('dwm-restored-from-sharpen.png', dwm_g)
# imshow(dwm_g)
# plt.show()
#
# # TEST 4.1: check simple inc and ext after sharpen (YUM, i.e only Y channel)
# #
# image, dwm = imread('tiger-color.png'), imread('dwm2.bmp')
#
# sharp_mask = 1 / 8 * np.array([[-1, -1, -1],
#                                [-1, 16, -1],
#                                [-1, -1, -1]])[:, :]
#
# im_with_dwm = insert_dwm(image, dwm)
# dwm_layer = extract_dwm(im_with_dwm)
# imshow(dwm_layer)
# plt.show()
# im_with_dwm = convolution_yuv(im_with_dwm.copy(), sharp_mask)
# imsave('cont-with-dwm-sharp(yuv).png', im_with_dwm)
# again_img = imread('cont-with-dwm-sharp(yuv).png')
# imshow(again_img)
# plt.show()
# dwm_layer = extract_dwm(again_img)
# imsave('dwm-layer-sharpen(yuv).png', dwm_layer)
# imshow(dwm_layer)
# plt.show()
# dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
# imsave('dwm-restored-from-sharpen.png', dwm_g)
# imshow(dwm_g)
# plt.show()

# TEST 4.2: check simple inc and ext after linear smoothin'
#
# image, dwm = imread('palm-tree.png'), imread('dwm3.bmp')
# mask = 1 / 21 * np.array([[2, 3, 2],
#                          [3, 1, 3],
#                          [2, 3, 2]])[:, :]
#
# # im_with_dwm = insert_dwm(image, dwm)
# im_with_dwm = convolution_yuv(image.copy(), mask)
# imsave('palm-tree-linear-smooth.png', im_with_dwm)
# imshow(im_with_dwm)
# plt.show()
# dwm_layer = extract_dwm(im_with_dwm)
# imshow(dwm_layer)
# plt.show()
# dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
# imshow(dwm_g)
# plt.show()
# print('Similarity = %.3f' % similarity(dwm_g, dwm))
# imsave('dwm-guess3.png', dwm_g)


# TEST 5: check simple inc and ext after median filter
#
image, dwm = imread('red-flower.png'), imread('dwm2.bmp')
im_with_dwm = insert_dwm(image, dwm)
dwm_layer = extract_dwm(im_with_dwm)
imshow(dwm_layer)
plt.show()
im_with_dwm = rng_filter_yuv(median, im_with_dwm.copy())
imsave('cont-with-dwm-median.png', im_with_dwm)
again_img = imread('cont-with-dwm-median.png')
imshow(again_img)
plt.show()
dwm_layer = extract_dwm(again_img)
imsave('dwm-layer-median.png', dwm_layer)
imshow(dwm_layer)
plt.show()
dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
imsave('dwm-restored-from-median.png', dwm_g)
imshow(dwm_g)
plt.show()

# TEST 6: check simple inc and ext after dilatation and erosion filters
#
image, dwm = imread('tiger-color.png'), imread('dwm2.bmp')
im_with_dwm = insert_dwm(image, dwm)
dwm_layer = extract_dwm(im_with_dwm)
imshow(dwm_layer)
plt.show()
im_with_dwm_eros = rng_filter_yuv(erosion, im_with_dwm.copy())
imsave('cont-with-dwm-erosion(yuv).png', im_with_dwm_eros)
again_img = imread('cont-with-dwm-erosion(yuv).png')
imshow(again_img)
plt.show()
dwm_layer = extract_dwm(again_img)
imsave('dwm-layer-erosion(yuv).png', dwm_layer)
imshow(dwm_layer)
plt.show()
dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
imsave('dwm-restored-from-erosion.png', dwm_g)
imshow(dwm_g)
plt.show()
im_with_dwm_dil = rng_filter_yuv(dilatation, im_with_dwm.copy())
imsave('cont-with-dwm-dilatation(yuv).png', im_with_dwm_dil)
again_img = imread('cont-with-dwm-dilatation(yuv).png')
imshow(again_img)
plt.show()
dwm_layer = extract_dwm(again_img)
imsave('dwm-layer-dilatation(yuv).png', dwm_layer)
imshow(dwm_layer)
plt.show()
dwm_g = dwm_guess(dwm_layer, dwm.shape[0], dwm.shape[1])
imsave('dwm-restored-from-dilatation.png', dwm_g)
imshow(dwm_g)
plt.show()
