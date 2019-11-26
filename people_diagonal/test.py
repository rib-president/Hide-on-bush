from PIL import Image
import os
import diagonal_crop




def recrop_image(image):
    img = image.convert('RGBA')
    pixeldata = list(img.getdata())
    width, height = img.size
    x_list = []
    y_list = []
    for i, pixel in enumerate(pixeldata):
        if pixel[3:4] == (255,):
            y, x = divmod(i, width)
            x_list.append(x)
            y_list.append(y)
    if len(x_list) > 0 or len(y_list) > 0:
        maxX = max(x_list)
        maxY = max(y_list)
        minX = min(x_list)
        minY = min(y_list)
        recropped_img = img.crop((minX, minY, maxX, maxY))
    else:
        recropped_img = img.crop((0, 0, 0, 0))
    return recropped_img





path = './image/'

img_list = sorted(os.listdir(path))

angle = 99.7

dst = './result/'
base = (1050, 250)

h = 600

w = 250

i=0

threshold = 10

for img in img_list:
    img_path = os.path.join(path, img)
    im = Image.open(img_path)
    crop_img = diagonal_crop.crop(im, base, angle, h, w)
    ccrop_img = recrop_image(crop_img)
    bg = Image.new('RGBA', ccrop_img.size, (0,0,0,0))
    bg.paste(ccrop_img)
    bg.save('{}/{:06d}.jpg'.format(dst, i), 'JPEG')
    i += 1



