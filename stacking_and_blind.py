# impulse fusion disabler and blind_embed
import PIL.Image as Image
from fusions import fusion_stacking
from additions import draw_graph_and_save, enoise_image_mul
from watermarks import embed_wm_blind_multi, extract_wmark_blind_multi, blind_extract_result
import numpy as np


def cycle(img_name: str, fus_type: str, dens: float, bits_am: int, amount_of_images: int, orig_image: Image.Image):

    images = []
    message = list(np.random.randint(2, size=bits_am))
    secret_key = 'secret'
    for i in range(amount_of_images):
        next_image = enoise_image_mul(0.15, orig_image)
        next_image = embed_wm_blind_multi(8, message, secret_key, next_image)
        images.append(next_image)

    images[0].save('outputs/'+img_name+'_stack_example('+str(dens)+'dens,'+fus_type+'_fustype,'+str(amount_of_images)+'pcs).bmp')
    restored_image = fusion_stacking(fus_type, images)
    restored_image.save('outputs/'+img_name+'_stack_restored('+str(dens)+'dens,'+fus_type+'_fustype,'+str(amount_of_images)+'pcs).bmp')
    print('extracting from restored...')
    ext_message = extract_wmark_blind_multi(.2, len(message), secret_key, restored_image)
    print('было: ', message)
    print('стало: ', ext_message)
    return dens, blind_extract_result(message, ext_message)


folder = 'examples/'
image_name = 'car-fast'
original: Image.Image = Image.open(folder + image_name + '.bmp')

bits_amount = 10
start = 0.1
one_step = 0.2
steps = 10

# test one: 5 images and mean
img_amount = 5
x_vector, y_vector = [], []
stack_type = 'mean'
for iteration in range(steps):
    x, y = cycle(image_name, stack_type, start + iteration * one_step, bits_amount, img_amount, original)
    x_vector.append(x)
    y_vector.append(y)
draw_graph_and_save('results/stack_fusion(mean)_and_blind_5_imgs', x_vector, y_vector, 'Зашумленность перед стэкингом %', 'Извлекаемость в %')
print('стэкинг и блайнд: первый тест отработан!')

# test two: 5 images and median
img_amount = 5
x_vector, y_vector = [], []
stack_type = 'median'
for iteration in range(steps):
    x, y = cycle(image_name, stack_type, start + iteration * one_step, bits_amount, img_amount, original)
    x_vector.append(x)
    y_vector.append(y)
draw_graph_and_save('results/stack_fusion(median)_and_blind_5_imgs', x_vector, y_vector, 'Зашумленность перед стэкингом %', 'Извлекаемость в %')
print('стэкинг и блайнд: второй тест отработан!')


# test three: 50 images and mean
img_amount = 50
x_vector, y_vector = [], []
stack_type = 'mean'
for iteration in range(steps):
    x, y = cycle(image_name, stack_type, start + iteration * one_step, bits_amount, img_amount, original)
    x_vector.append(x)
    y_vector.append(y)
draw_graph_and_save('results/stack_fusion(mean)_and_blind_50_imgs', x_vector, y_vector, 'Зашумленность перед стэкингом %', 'Извлекаемость в %')
print('стэкинг и блайнд: третий тест отработан!')


# test four: 50 images and median
img_amount = 50
x_vector, y_vector = [], []
stack_type = 'median'
for iteration in range(steps):
    x, y = cycle(image_name, stack_type, start + iteration * one_step, bits_amount, img_amount, original)
    x_vector.append(x)
    y_vector.append(y)
draw_graph_and_save('results/stack_fusion(median)_and_blind_50_imgs', x_vector, y_vector, 'Зашумленность перед стэкингом %', 'Извлекаемость в %')
print('стэкинг и блайнд: четвертый тест отработан!')
