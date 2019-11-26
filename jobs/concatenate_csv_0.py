import os


f_list = ['train']
#f_list = ['train']

for f in f_list:
    fout=open("./"+str(f)+"_cali_7_3.csv","a")

    file_name = f +"_2178_minus_13.csv"    
#        print(file_name)
    for line in open(file_name):        
        fout.write(line)    
    bfile_name = "./mlp_data_cali_7_3.csv"    
#        print(file_name)
    for lline in open(bfile_name):        
        fout.write(lline)    





fout.close()
