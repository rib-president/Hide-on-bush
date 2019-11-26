from PIL import Image
import os


img_path = "../maskrcnn/"
save_root_path = "../maskrcnn_fin/"
folders_list = sorted(os.listdir(img_path))

def filecheck(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print ('create' + path)
        except OSError as e:
            if e.errno != e.errno.EEXIST:
                raise

for folder_list in folders_list:
    folder_path = os.path.join(img_path,folder_list)
    barcodes_list = sorted(os.listdir(folder_path))
    for barcode_list in barcodes_list:
        barcode_path = os.path.join(folder_path,barcode_list)
        imgs_list = sorted(os.listdir(barcode_path))
        save_path = os.path.join(os.path.join(save_root_path , folder_list) , barcode_list)
        if os.path.exists(save_path):
            print(save_path + " exists")
            continue
        filecheck(save_path)
        for img_list in imgs_list:
            image=Image.open(os.path.join(barcode_path,img_list))
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

            recropped_img.save(save_path+'/'+barcode_list+"_"+img_list.split('.')[0]+'.png')
