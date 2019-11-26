import os
import shutil



val = './iphone_2296'
train = './train/'

train_list = os.listdir(train)

folder_list = os.listdir(val)

for folder in folder_list:
    folder_path = os.path.join(val, folder)
    if folder in train_list:
        img_list = os.listdir(folder_path)
        for img in img_list:
            img_path = os.path.join(folder_path, img)
            name = os.path.basename(img_path)        
            os.rename(img_path, train  + folder + '/'  + name)
        shutil.rmtree(folder_path)

