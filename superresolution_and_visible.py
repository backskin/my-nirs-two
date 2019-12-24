import PIL.Image as Image
from additions import draw_graph_and_save
from watermarks import embed_wm_dct
from fusions import fusion_superresolution, scan_superresolution
import numpy as np

folder = 'examples/'
image_name = 'car-jeep'
original: Image.Image = Image.open(folder + image_name + '.bmp')
image_wm: Image.Image = Image.open(folder + 'wm.bmp')
bits_am = 10
message = list(np.random.randint(2, size=bits_am))
secret_key = 'secret'

images = scan_superresolution(4, original)
embed_wm_dct(original, image_wm).save('results/SR_vis_orig_example.bmp')
for i in range(len(images)):
    for j in range(len(images[i])):
        images[i][j] = embed_wm_dct(images[i][j], image_wm)

restored_img = fusion_superresolution(images)
restored_img.save('results/SR_vis_2.bmp')