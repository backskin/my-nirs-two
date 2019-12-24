# impulse fusion disabler and blind_embed
import PIL.Image as Image
from fusions import fusion_impulse_filter
from additions import get_imp_noise, draw_graph_and_save
from watermarks import embed_wm_blind_multi, extract_wm_blind_multi, blind_extract_result
import numpy as np


def cycle(img_name: str, dens: float, bits_am: int, amount_of_images: int, orig_image: Image.Image):
    images = []
    message = list(np.random.randint(2, size=bits_am))
    secret_key = 'secret'
    for i in range(amount_of_images):
        next_image = embed_wm_blind_multi(0.1, message, secret_key, orig_image)
        next_image = get_imp_noise(dens, next_image)
        images.append(next_image)
    images[0].save('outputs/'+img_name+'_imp_noise_example('+str(dens)+'dens,'+str(amount_of_images)+'pcs).bmp')
    restored_image = fusion_impulse_filter(images)
    restored_image.save('outputs/'+img_name+'_imp_noise_restored('+str(dens)+'dens,'+str(amount_of_images)+'pcs).bmp')
    print('extracting from restored...')
    ext_message = extract_wm_blind_multi(.2, len(message), secret_key, restored_image)
    print('было: ', message)
    print('стало: ', ext_message)
    return dens, blind_extract_result(message, ext_message)


folder = 'examples/'
image_name = 'pineapple'
original: Image.Image = Image.open(folder + image_name + '.bmp')

bits_amount = 10
start = 0.1
one_step = 0.1
steps = 10

# test one: 5 images
img_amount = 5
x_vector, y_vector = [], []
for iteration in range(steps):
    x, y = cycle(image_name, start + iteration * one_step, bits_amount, img_amount, original)
    x_vector.append(x*100)
    y_vector.append(y*100)
draw_graph_and_save('results/imp_fusion_and_blind_5_imgs', x_vector, y_vector, 'Зашумленность в %', 'Извлекаемость в %')
print('импульс и блайнд: первый тест отработан!')

# test two: 20 images
img_amount = 20
x_vector, y_vector = [], []
for iteration in range(steps):
    x, y = cycle(image_name, start + iteration * one_step, bits_amount, img_amount, original)
    x_vector.append(x)
    y_vector.append(y)
draw_graph_and_save('results/imp_fusion_and_blind_20_imgs', x_vector, y_vector, 'Зашумленность в %', 'Извлекаемость в %')
print('импульс и блайнд: второй тест отработан!')
