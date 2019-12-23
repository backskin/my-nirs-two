# impulse fusion disabler and blind_embed
import PIL.Image as Image
from fusions import fusion_impulse_filter
from additions import get_imp_noise
from watermarks import embed_wmark_blind_multi, extract_wmark_blind_multi
import numpy as np


def cycle(dens: float, bits_am: int, amount_of_images: int, orig_image:Image.Image):
    images = []
    message = list(np.random.randint(2, size=bits_am))
    secret_key = 'secret'
    for i in range(amount_of_images):
        print(i, '-s iteration in noise/embedding')
        next_image = get_imp_noise(dens, orig_image)
        next_image = embed_wmark_blind_multi(8, message, secret_key, next_image)
        images.append(next_image)

    restored_image = fusion_impulse_filter(images)
    print('extracting from restored...')
    ext_message = extract_wmark_blind_multi(.2, len(message), secret_key, restored_image)
    print('было: ', message)
    print('стало: ', ext_message)


folder = 'examples/'
image_name = 'pineapple'
original: Image.Image = Image.open(folder + image_name + '.bmp')

# test one: 5 images
img_amount = 5
bits_amount = 10
start = 0.1
one_step = 0.1
steps = 10
for iteration in range(steps):
    cycle(start + iteration * one_step, bits_amount, img_amount, original)

# test two: 20 images
img_amount = 20
bits_amount = 10
start = 0.1
one_step = 0.1
steps = 10
for iteration in range(steps):
    cycle(start + iteration * one_step, bits_amount, img_amount, original)