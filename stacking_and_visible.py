# stacking gauss noise disabler and visible watermark (my_own_method)
import PIL.Image as Image
from additions import enoise_image_mul
from fusions import fusion_stacking
from watermarks import embed_wm_dct
import matplotlib.pyplot as plt


def cycle(img_name: str, fus_type: str, amount_of_images: int,  wm_image: Image.Image, orig_image: Image.Image):
    images = []

    for i in range(amount_of_images):
        next_image = enoise_image_mul(0.15, orig_image)
        next_image = embed_wm_dct(next_image, wm_image)
        images.append(next_image)

    images[0].save(img_name+'_visible_wm_example('+str(amount_of_images)+').bmp')
    restored_image = fusion_stacking(fus_type, images)
    restored_image.save(img_name+'_visible_wm_restored('+str(amount_of_images)+').bmp')


folder = 'examples/'
image_name = 'car-jeep'
original: Image.Image = Image.open(folder + image_name + '.bmp')
watermark: Image.Image = Image.open(folder + 'wm.bmp')
results_folder = 'results/'


# test one: 10 images and mean
images_amount = 10
cycle(results_folder+image_name, 'mean', images_amount, watermark, original)

# test two: 50 images and mean
images_amount = 50
cycle(results_folder+image_name, 'mean', images_amount, watermark, original)