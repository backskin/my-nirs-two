import PIL.Image as Image
from additions import draw_graph_and_save
from watermarks import embed_wm_dm_qim, extract_wm_dm_qim
from fusions import fusion_two_images



folder = 'examples/'
image_name = 'pineapple'
sec_name = 'sunflowers'
original: Image.Image = Image.open(folder + image_name + '.bmp')
second: Image.Image = Image.open(folder + sec_name + '.bmp')
watermark: Image.Image = Image.open(folder + 'wm.bmp')
secret_key = 'lilly'

second = embed_wm_dm_qim(secret_key, 16, second, watermark)

fused = fusion_two_images(original, 0, 0, 0.4, second, 0, 0, 0.7)
fused.save('fused_'+image_name+'_'+sec_name+'_0.4_0.7.bmp')
ext_wm = extract_wm_dm_qim(secret_key, 16, fused)
ext_wm.save('alpha_wm_0.5_0.5_extracted.bmp')
