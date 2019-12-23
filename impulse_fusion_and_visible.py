# impulse fusion disabler and visible watermark (additive)
import PIL.Image as Image
from watermarks import embed_wmark_additive
from additions import get_imp_noise
from fusions import fusion_impulse_filter


def cycle(name: str, dens: float, amount_of_images: int, wm_image: Image.Image, orig_image: Image.Image):
    images = []
    for i in range(amount_of_images):
        next_image = get_imp_noise(dens, orig_image)
        next_image = embed_wmark_additive(4, next_image, wm_image)
        images.append(next_image)

    restored_image = fusion_impulse_filter(images)
    restored_image.save(name + '_restored_with_visual_watermark.bmp')


folder = 'examples/'
out_folder = 'outputs/'
image_name = 'pineapple'
original: Image.Image = Image.open(folder + image_name + '.bmp')
watermark: Image.Image = Image.open(folder + 'wm.bmp')

# test: 20 images
img_amount = 20
start = 0.1
one_step = 0.1
steps = 10
for iteration in range(steps):
    cycle(out_folder+image_name, start + iteration * one_step, img_amount, original, watermark)
