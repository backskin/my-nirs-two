import PIL.Image as Image
from watermarks import embed_wm_blind_multi, extract_wm_blind_multi, embed_wm_dm_qim, extract_wm_dm_qim, qim_extract_result
from additions import enoise_image_mul as noise_equal
from fusions import fusion_stacking, fusion_superresolution, scan_superresolution
import matplotlib.pyplot as plt
import os


folder = 'examples/'
out_folder = 'outputs/'
noise_folder = 'noises/'
if not os.path.exists(folder):
    os.mkdir(folder)
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
if not os.path.exists(noise_folder):
    os.mkdir(noise_folder)

image_name = 'sunflowers'
original: Image.Image = Image.open(folder + image_name + '.bmp')
watermark: Image.Image = Image.open(folder + 'wm_inverse.bmp')

secret_key = 'kitty'



# images = scan_superresolution(4, original)
# for i in range(len(images)):
#     for j in range(len(images[i])):
#         print((i, j))
#         # images[i][j] = embed_wm_dm_qim(secret_key, 8, original, watermark)
# print('fusing_SR...')
# res_img = fusion_superresolution(images)
# res_img.save(image_name+'_SR_result.bmp')
# print('extracting_wm...')
# # wm_extracted = extract_wm_dm_qim(secret_key, 8, res_img)
# plt.imshow(wm_extracted)
# plt.show()
# p = qim_extract_result(wm_extracted, watermark)
# print(p)

# amount = 50
# std_dev = .5
# noise_list = gauss_noisy_shots(image_name, amount, std_dev, one)
# stack_type = 'median'
# restored_image = fusion_stacking(stack_type, noise_list)
# restored_image.save(out_folder+image_name+'_'+str(amount)+'-pcs_'+stack_type+'-type_restored_from_gauss_noise.bmp')

# image_name = 'car-jeep'
# wm_name = 'wm'
# one: Image.Image = Image.open(folder + image_name + '.bmp')
# key = 'hello'
# bit_mes = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
# emb_img = embed_wm_blind_multi(8, bit_mes, key, one)
# emb_img.save(out_folder+image_name+'_with_blind_multi_wm('+key+').bmp')
# print('embeded')
# b_res = extract_wm_blind_multi(0.2, len(bit_mes), key, emb_img)
# print('extracted')
# print(b_res)


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
# three = embed_wm_dct(one, two)
# # three = embed_wm_dm_qim(key, delta, one, two)
# three.save(image_name+'_with_dct_'+wm_name+'.bmp')
# print('extracting...')
# extracted = extract_wm_dct(two.size, three)
# # extracted = extract_wm_dm_qim(key, delta, three)
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
# t_frames = scan_superresolution(8, orig)
# print('matsynth started')
# new_image = mat_synth(t_frames)
# new_image.save(image_name+'_matrix.bmp')
