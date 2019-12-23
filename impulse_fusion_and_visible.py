# impulse fusion disabler and visible watermark (additive)
import PIL.Image as Image
from watermarks import embed_wmark_additive
from additions import get_imp_noise
from fusions import fusion_impulse_filter


def cycle(name: str, dens: float, amount_of_images: int, orig_image: Image.Image, wm_image: Image.Image):
    images = []
    for i in range(amount_of_images):
        next_image = get_imp_noise(dens, orig_image)
        next_image = embed_wmark_additive(0.09, next_image, wm_image)
        images.append(next_image)
    print('намутили для дэнс = ', dens)
    restored_image = fusion_impulse_filter(images)
    restored_image.save(name + '_('+str(dens)+'dens,'+str(amount_of_images)+'pcs.)_restored_with_visual_watermark.bmp')


folder = 'examples/'
out_folder = 'outputs/'
image_name = 'statue'
original: Image.Image = Image.open(folder + image_name + '.bmp')
watermark: Image.Image = Image.open(folder + 'wm_inverse.bmp')

# test: 20 images
img_amount = 10
start = 0.5
one_step = 0.1
steps = 5
for iteration in range(steps):
    cycle(out_folder+image_name, start + iteration * one_step, img_amount, original, watermark)
