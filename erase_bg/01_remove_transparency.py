from PIL import Image
import os


img_path = "contour"
save_root_path = "no_transparency"
folders_list = sorted(os.listdir(img_path))

def filecheck(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print ('create' + path)
        except:
            pass

for folder_list in folders_list:
    folder_path = os.path.join(img_path,folder_list)
    imgs_list = sorted(os.listdir(folder_path))
    save_path = os.path.join(save_root_path , folder_list)

    filecheck(save_path)
    for img_list in imgs_list:
        image=Image.open(os.path.join(folder_path,img_list))
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

        recropped_img.save(save_path+'/'+ img_list.split('.')[0]+'.png')
