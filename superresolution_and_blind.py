import PIL.Image as Image
from additions import draw_graph_and_save
from watermarks import embed_wm_blind_multi, extract_wm_blind_multi, blind_extract_result
from fusions import fusion_superresolution, scan_superresolution
import numpy as np

folder = 'examples/'
image_name = 'car-jeep'
original: Image.Image = Image.open(folder + image_name + '.bmp')
# image_wm: Image.Image = Image.open(folder + 'wm.bmp')
bits_am = 10
message = list(np.random.randint(2, size=bits_am))
secret_key = 'secret'

# images = scan_superresolution(16, original)
#
# for i in range(len(images)):
#     for j in range(len(images[i])):
#         images[i][j] = embed_wm_blind_multi(8, message, secret_key, images[i][j])
#
# restored_img = fusion_superresolution(images)
#
# b_ar = extract_wm_blind_multi(.1, bits_am, secret_key, restored_img)
# p = blind_extract_result(b_ar, message)
# print(message, b_ar)
# print(p)