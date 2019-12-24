# fusion superresolution and qim_embed
import PIL.Image as Image
from additions import draw_graph_and_save
from watermarks import embed_wm_dm_qim, extract_wm_dm_qim, qim_extract_result
from fusions import fusion_superresolution, scan_superresolution


def cycle(img_name: str, scale: int, watermark: Image.Image, orig_image: Image.Image):

    images = scan_superresolution(scale, orig_image)
    secret_key = 'secret'
    for i in range(len(images)):
        for j in range(len(images[i])):
            images[i][j] = embed_wm_dm_qim(secret_key, 16, images[i][j], watermark)

    images[0][0].resize(orig_image.size, Image.NONE).save('outputs/'+img_name+'_SR_example('+str(scale)+'_scale).bmp')
    restored_image = fusion_superresolution(images)
    restored_image.save('outputs/'+img_name+'_SR_restored('+str(scale)+'_scale).bmp')
    print('extracting from restored...')
    extracted_wm = extract_wm_dm_qim(secret_key, 16, restored_image)
    extracted_wm.save('outputs/'+img_name+'_SR_wm_extracted('+str(scale)+'_scale).bmp')
    return qim_extract_result(extracted_wm, watermark)


folder = 'examples/'
image_name = 'car-fast'
original: Image.Image = Image.open(folder + image_name + '.bmp')
image_wm: Image.Image = Image.open(folder + 'wm.bmp')
bits_amount = 10
start = 0.1
one_step = 0.1
steps = 10
x_vector, y_vector = [2,4,8,16], []
draw_graph_and_save('results/SR_fusion_and_qim', x_vector, y_vector, 'Сжатие (в k раз)', 'Извлекаемость в %')

# test one: 2 times smaller
zip_scale = 1
y = cycle(image_name, zip_scale, image_wm, original)
# x_vector.append(zip_scale*zip_scale)
# y_vector.append(y)
print('суперразрешение и квим: первый тест отработан!')
#
# # test two: 4 times smaller
# zip_scale = 4
# y = cycle(image_name, zip_scale, image_wm, original)
# x_vector.append(zip_scale*zip_scale)
# y_vector.append(y)
# print('суперразрешение и квим: второй тест отработан!')
#
# # test three: 8 times smaller
# zip_scale = 8
# y = cycle(image_name, zip_scale, image_wm, original)
# x_vector.append(zip_scale*zip_scale)
# y_vector.append(y)
# print('суперразрешение и квим: третий тест отработан!')
#
# # test four: 16 times smaller
# zip_scale = 16
# y = cycle(image_name, zip_scale, image_wm, original)
# x_vector.append(zip_scale*zip_scale)
# y_vector.append(y)
# print('суперразрешение и квим: четвертый тест отработан!')
#
# # test five: 24 times smaller
# zip_scale = 24
# y = cycle(image_name, zip_scale, image_wm, original)
# x_vector.append(zip_scale*zip_scale)
# y_vector.append(y)
# print('суперразрешение и квим: пятый тест отработан!')
