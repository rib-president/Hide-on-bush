import os
import shutil
import sys
import csv
import argparse


def split_more(root_path, dst, move_num, float_num):
    for fl in range(float_num):
        folder_list = sorted(os.listdir(root_path))
        move_path = os.path.join(dst, '%02d' %fl)
        if not os.path.exists(move_path):
            os.makedirs(move_path)
            
        for i in range(int(move_num)+1):
            folder_path = os.path.join(root_path, folder_list[i])
            #shutil.copytree(folder_path, move_path + '/' + folder_list[i])
            shutil.move(folder_path, move_path)
            
            #make_mapping(folder_list[i])
            
    return fl                    
    


def split_common(root_path, divide_num, move_num, folder_num):    
    folder_list = sorted(os.listdir(root_path))
    cnt = 0
    print 'len', len(folder_list)
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
            
        #shutil.copytree(folder_path, move_path + '/' + folder)
        shutil.move(folder_path, move_path)



def merge(folder_path):
    real_path = os.path.abspath(folder_path)
    forePath = '/'.join(real_path.split('/')[:-1])
    backPath = real_path.split('/')[-1].split('_')[1:]
    return_path = os.path.join(forePath, '_'.join(backPath))
    #return_path = '_'.join(folder_path.split('_')[1:])
    if not os.path.exists(return_path):
        os.makedirs(return_path)
    split_folder_list = sorted(os.listdir(folder_path))
    for split_folder in split_folder_list:
        split_folder_path = os.path.join(folder_path, split_folder)
        barcode_list = sorted(os.listdir(split_folder_path))
        for barcode in barcode_list:
            barcode_path = os.path.join(split_folder_path, barcode)
            shutil.move(barcode_path, return_path)
            #os.makedirs(barcode_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        metavar="<command>",
                        help="'split' or 'merge'")
    parser.add_argument('--total_num', required=False,
                        metavar="ex)60",
                        help='Total number of folders')
    parser.add_argument('--divide_num', required=False,
                        metavar="total_num / divide_num",
                        help='number of how many folder will split.')
    parser.add_argument('--root', required=False,
                        metavar="/path/to/folder/",
                        help="root folder that will be 'split' or 'merge'")
                    
    args = parser.parse_args()                                                                     

    if args.command == "split":
        assert args.total_num, "Argument --total_num is required for split"
        assert args.divide_num, "Argument --divide_num is required for split"


        #move_num = int(args.total_num) / float(args.divide_num)
        move_num = int(args.total_num) / int(args.divide_num)
        total_num = int(args.total_num)
        divide_num = int(args.divide_num)        
        
        
        if '/' == args.root[-1]: 
            root_path = args.root[:-1]
        else:
            root_path = args.root
        forePath = '/'.join(root_path.split('/')[:-1]) 
        dst = os.path.join(forePath, 'split_'+os.path.basename(root_path))
        if not os.path.exists(dst):
            os.makedirs(dst)



        #float_num = int((move_num - int(move_num))*10)
        float_num = int(args.total_num) % int(args.divide_num)
        if float_num > divide_num:
            float_num = float_num / 2
        if float_num > 0:
            last_folder = split_more(root_path, dst, move_num, float_num)
            split_common(root_path, divide_num, move_num, last_folder+1)
    
        else:
            split_common(root_path, divide_num, move_num, 0)

    elif args.command == "merge":
        merge(args.root)
        shutil.rmtree(args.root)
