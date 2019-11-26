import os
import shutil
import sys
import csv


tot_num = int(sys.argv[1])
divide_num = float(sys.argv[2])
root_path = sys.argv[3]

move_num = tot_num / divide_num

dst = os.path.join('./', 'split_'+os.path.basename(root_path))
if not os.path.exists(dst):
    os.makedirs(dst)

### set mapping file name
mapping_file_name = './mapping.csv'


### for added products training
try:
    with open('%s' % mapping_file_name, 'r') as df:
        read = csv.reader(df, delimiter=',')
        for row in read:
            start_idx = int(row[0])+1

except:
    start_idx = 0 

      

def split_more(float_num):
    for fl in range(float_num+1):
        folder_list = sorted(os.listdir(root_path))
        move_path = os.path.join(dst, '%02d' %fl)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
            
        for i in range(int(move_num)+1):
            folder_path = os.path.join(root_path, folder_list[i])
            shutil.move(folder_path, move_path)
            make_mapping(folder_list[i])
            
    return fl                    
    


def split_common(folder_num):    
    folder_list = sorted(os.listdir(root_path))
    cnt = 0
    for folder in folder_list:
        folder_path = os.path.join(root_path, folder)

        if cnt >= int(move_num):    
            cnt = 0
            folder_num += 1
        
        if not folder_num >= divide_num:
            cnt +=1
            
        else:
            folder_num -= 1
            
        move_path = os.path.join(dst, '%02d' %folder_num)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
            
        shutil.move(folder_path, move_path)
        make_mapping(folder)


def make_mapping(barcode):
    global start_idx    
    with open('%s' % mapping_file_name, 'a') as df:
        write = csv.writer(df, delimiter=',')
        write.writerow([start_idx, barcode])
        start_idx+=1
            


def merge(folder_path):
    return_path = '_'.join(folder_path.split('_')[1:])
    if not os.path.exists(return_path):
        os.makedirs(return_path)
    split_folder_list = sorted(os.listdir(folder_path))
    for split_folder in split_folder_list:
        split_folder_path = os.path.join(folder_path, split_folder)
        barcode_list = sorted(os.listdir(split_folder_path))
        for barcode in barcode_list:
            barcode_path = os.path.join(split_folder_path, barcode)
            shutil.move(barcode_path, return_path)
            os.makedirs(barcode_path)

'''
if __name__ == '__main__':
    merge('./split_test_1')
'''           

if __name__ == '__main__':
    float_num = int((move_num - int(move_num))*10)
       
    if float_num > 0:
        last_folder = split_more(float_num)
        split_common(last_folder+1)
    
    else:
        split_common(0)

