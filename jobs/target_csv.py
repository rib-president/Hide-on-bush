import os
import csv
'''
#f_list = ['train','val']
f_list = ['test']

for f in f_list:
    fout=open(str(f)+".csv","a")
    for num in range(0,12):
        file_name = "mlp_data_barcode_"+str(f)+"_"+str("%02d"%(num))+".csv"    
#        print(file_name)
        for line in open(file_name):        
            fout.write(line)    

fout.close()
'''
csv_path = './train_6_4_uptodate.csv'

def read_csv():
    mycsv = csv.reader(open(csv_path))
    label_list = []
    for row in mycsv:
        label = row[0]
#        if not str(label) in ['349','272','325','22','178']:
        if not str(label) in ['900', '1058', '1059']:
        
            print(str(label))
            with open('train_6_4_uptodate_new.csv', 'a') as f:
                w = csv.writer(f, delimiter=',')
                w.writerow(row)
                f.close()      
            

read_csv()
