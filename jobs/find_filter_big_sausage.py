import os
import csv
import ast
import sys


cutLine = int(sys.argv[1])
remainder = cutLine - 1




def chase_point(master, stat):
    count = 0
    grid_dict = {}
    final_dict = {}
    filter_dict = {}
    l = 0
    long_too = []
                
    filter_list = []
    filter_list.append(master)
    for most in stat[0]:
        most = ast.literal_eval(most)
        filter_list.append(most)

    with open('inverse_filter_confused_5product_big_sausage.csv', 'a') as inf:
        inf_w = csv.writer(inf, delimiter=',')
        inf_w.writerow(filter_list)    
    
    
    

if __name__ == '__main__':
    tmp = []
    stat = []
    cnt = 0
    with open('statistics_mlp_sk_test_1000_confused_5product.csv', 'r') as df:
        read = csv.reader(df, delimiter=',')
        for i, row in enumerate(read):
            if i % cutLine != remainder:
                if row[4] == '1':
                    tmp.append(row[3])
                    cnt += 1

                master = row[0]
                stat.append(tmp)
            else:
                if len(stat) != 0:
                    chase_point(master,stat)
                else:
                    print '=========================================================================='
                    print 'can not find proper filter, check %s' % master
                    with open('check_noFilter.csv', 'a') as ddf:
                        write = csv.writer(ddf, delimiter=',')
                        write.writerow([master])
                stat = []
                tmp = []
                cnt = 0
                if row[4] == '1':
                    tmp.append(row[3])
                    cnt += 1

