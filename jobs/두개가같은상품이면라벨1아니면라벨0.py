import os
import glob
import shutil
import csv
import random
from datetime import datetime


csv_path = './test.csv'
data_dict = {}

start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S:%f')[:-3]

def read_csv():
    mycsv = csv.reader(open(csv_path))
    now_label = []
    cnt = 10
    for row in mycsv:
        if len(now_label) == 0:        
            now_label.append(row[0])
        else:
            if now_label[0] == row[0]:
                pass
            else:
                now_label = []
                cnt =10
        label = row[0] + '_'+str(cnt)
        float_list = []
        for r in row:
            if float(r) <100:
                float_list.append(r)
        data_dict[label] = float_list 
        cnt -=1


def make_1(key):
    cnt = 0
    barcode_list = sorted(data_dict.keys())
    barcode_list.remove(key)
    
    for barcode in barcode_list:
        tmp_list = []
        if barcode.split('_')[0] == key.split('_')[0]:
            tmp_list = tmp_list + ['1'] + data_dict[key] + data_dict[barcode]
            write_csv(tmp_list)
            cnt += 1
    return cnt
            


def make_0(key, max_iter):
    cnt = 0
    barcode_list = sorted(data_dict.keys())
    barcode_list.remove(key)
    
    while cnt < max_iter:
        tmp_list = []
        barcode = random.choice(barcode_list)
        if barcode.split('_')[0] != key.split('_')[0]:
            tmp_list = tmp_list + ['0'] + data_dict[key] + data_dict[barcode]
            write_csv(tmp_list)
            cnt += 1

            

def write_csv(tmp_list):
    with open('final_csv.csv', 'a') as df:
        write = csv.writer(df, delimiter=',')
        write.writerow(tmp_list)



if __name__ == '__main__':    
    read_csv()
    
    for key in sorted(data_dict.keys()):
        cnt = make_1(key)
        make_0(key, cnt)
        
print 'start', start_time, 'end', datetime.now().strftime('%Y/%m/%d %H:%M:%S:%f')[:-3]
