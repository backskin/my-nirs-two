import PIL.Image as Image
from additions import draw_graph_and_save, enoise_image_mul
from watermarks import embed_wm_dm_qim, extract_wm_dm_qim, qim_extract_result, embed_wm_dct, extract_wm_dct
from fusions import fusion_stacking


folder = 'examples/'
image_name = 'car-fast'
original: Image.Image = Image.open(folder + image_name + '.bmp')
image_wm: Image.Image = Image.open(folder + 'wm.bmp')
secret_key = 'secret'
images = []
amount_of_images = 10
for i in range(amount_of_images):
    next_image = enoise_image_mul(0.15, original)
    next_image = embed_wm_dct(next_image, image_wm)
    images.append(next_image)

restored_image = fusion_stacking('median', images)
extracted_wm = extract_wm_dct(image_wm.size, restored_image)
restored_image.save('stack_vis_rest.bmp')
extracted_wm.save('stack_vis_wm_ext.bmp')