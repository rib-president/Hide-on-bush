import os
import glob
root_list = ['train', 'val']
count_list = [(120, 160), (40, 30)]

for idx, root_name in enumerate(root_list):
    root_path = './' + root_name + '_*'
    root_dir = glob.glob(root_path)
    for root_idx, root in enumerate(root_dir):
        barcode_list = glob.glob(root + '/*')
        for barcode in barcode_list:
            if len(os.listdir(barcode)) is not count_list[idx][root_idx]:
                print barcode
