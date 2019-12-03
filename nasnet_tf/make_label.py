import os

# define your src_path
src_path = '/home/jung/nasnet-tensorflow/emart/'

folder_list = sorted(os.listdir(src_path))

for folder in folder_list :
    folder_path = os.path.join(src_path, folder)
    path, name = os.path.split(folder_path)
    with open('./demo/labelmap/label.txt', 'a') as f:
        f.write(name + '\n')


