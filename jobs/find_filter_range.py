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
    
    for st in stat:
        bg_list = []
        fg_list = []
        for sst in st:
            sst = ast.literal_eval(sst)
            fg_list.append(float(sst['fg']))
        fg_list = list(set(fg_list))
        fg_start = min(fg_list)
        fg_end = max(fg_list)
        x = fg_end - fg_start
        filter_dict['fg_start_%02d' % count] = str(fg_start)
        filter_dict['fg_end_%02d' % count] = str(fg_end)
        bg_start_list = []
        bg_end_list = []
        for sst in st:
            sst = ast.literal_eval(sst)
            if sst['fg'] == str(fg_start):
                bg_start_list.append(float(sst['bg']))
            elif sst['fg'] == str(fg_end):
                bg_end_list.append(float(sst['bg']))
        bg_start = min(list(set(bg_start_list)))
        bg_end = max(list(set(bg_end_list)))
        filter_dict['bg_start_%02d' % count] =str(bg_start)
        filter_dict['bg_end_%02d' % count] = str(bg_end)
        y = bg_end - bg_start
        grid = x*y
        
        grid_dict['grid_%02d' % count ] = grid
        
        count += 1




    maxGrid = max(grid_dict.values())

    for k, v in grid_dict.items():
        if v == maxGrid:
            maxGrid_idx = k.split('_')[-1]



    final_dict['fg_start_%s' % maxGrid_idx] = filter_dict['fg_start_%s' % maxGrid_idx]
    final_dict['fg_end_%s' % maxGrid_idx] = filter_dict['fg_end_%s' % maxGrid_idx]
    final_dict['bg_start_%s' % maxGrid_idx] = filter_dict['bg_start_%s' % maxGrid_idx]
    final_dict['bg_end_%s' % maxGrid_idx] = filter_dict['bg_end_%s' % maxGrid_idx]

    print '=========================================================================='
    print master, final_dict
    with open('inverse_filter.csv', 'a') as inf:
        inf_w = csv.writer(inf, delimiter=',')
        inf_w.writerow([master, final_dict])


if __name__ == '__main__':
    tmp = []
    stat = []
    cnt = 0
    with open('statistics_mlp_sk_test_1000_extreme.csv', 'r') as df:
        read = csv.reader(df, delimiter=',')
        for i, row in enumerate(read):
            if i % cutLine != remainder:
                if row[4] == '1':
                    tmp.append(row[3])
                    cnt += 1
                elif row[4] == '0' :
                    if cnt >= 10:
                        stat.append(tmp)
                    tmp = []
                    cnt = 0
                master = row[0]
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

