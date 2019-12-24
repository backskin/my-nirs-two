import PIL.Image as Image
from additions import draw_graph_and_save
from watermarks import embed_wm_dct, extract_wm_dct
from fusions import fusion_two_images


folder = 'examples/'
image_name = 'pineapple'
sec_name = 'sunflowers'
original: Image.Image = Image.open(folder + image_name + '.bmp')
second: Image.Image = Image.open(folder + sec_name + '.bmp')
watermark: Image.Image = Image.open(folder + 'wm.bmp')

second = embed_wm_dct(second, watermark)

fused = fusion_two_images(original, 0, 0, 0.8, second, 0, 0, 0.2)
fused.save('fused_vis_'+image_name+'_'+sec_name+'_0.8_0.2.bmp')