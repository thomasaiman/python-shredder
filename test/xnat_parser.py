# -*- coding: utf-8 -*-
"""
takes the xml file from the xnat subject page and creates a csv file and a
dictionary (stored as a pickle file) that details the relationships between
each slice and scn file
"""
import xmltodict
import csv
from copy import copy
import pickle

filepath = 'C:\\Users\\Thomas\\Downloads\\new_hazel.xml'

fileobj = open(filepath)
a = xmltodict.parse(fileobj.read())
fileobj.close()
b = dict()

c = a['xnat:Subject']['xnat:experiments']['xnat:experiment']
print("c = a['xnat:Subject']['xnat:experiments']['xnat:experiment']")
print('this is a list. showing keys for each entry')
count = 0
for exp in c:
    print('\n c[{}]:'.format(count))
    for k in exp.keys():
        if type(exp[k])==str:
            print(k,' = ', exp[k])
        else:
            print(k, ' = ', type(exp[k]))
    if 'histo' in exp['@label']:
        b[exp['@label']]=exp
        print('keep this one!')
    count += 1
    
d2=dict()
for k in b.keys():
    d = b[k]
    d1 = dict()
    e=d['xnat:scans']['xnat:scan']
    for scantype in e:
        filetype = scantype['@ID']
        if 'scn_files' in filetype: filetype='scn_files'
        elif 'hi_res_files' in filetype: filetype='hi_res_files'
        else: continue
        slicelist = list()
        for s in scantype['xnat:file']:
            slice_str = s['@label']
            slice_num = int(slice_str.split(sep='_')[1])
            slicelist.append(slice_num)
        d1[filetype] = slicelist
    
    e=d['xnat:assessors']['xnat:assessor']
    for assessortype in e:
        filetype = assessortype['@label']
        if 'df_files' in filetype:
            slicelist = list()
            for s in assessortype['xnat:out']['xnat:file']:
                slice_str = s['@label']
                if 'light_block3_01' in slice_str:
                    slice_num = int(slice_str.replace('-','_').split(sep='_')[1])
                    slicelist.append(slice_num)
            d1['df_files'] = slicelist
    d2[k]=d1

    
for stain in d2.keys():
    print(stain)
    csvname = stain+'.csv'
    picklename = stain+'.p'
    
    d3 = dict()
    fieldnames=('slice_num', 'df_files', 'scn_files', 'hi_res_files', 
                'scn_to_use', 'slices_in_this_scn')
    slicedict = {k:0 for k in fieldnames}
    for filetype,slicelist in d2[stain].items(): 
        for slice_num in slicelist:
            d3.setdefault(slice_num,copy(slicedict))
            d3[slice_num]['slice_num'] = slice_num
            d3[slice_num][filetype]=1
     
#    slicelist = list(d3.keys())
#    for k in slicelist:
#        if d3[k]['df_files'] == 0:
#            del d3[k]
            
    slicelist = sorted(d3.keys())
    scn_to_use = -1
    for k in slicelist:
        if d3[k]['scn_files'] == 1: scn_to_use = k
        if scn_to_use==-1: continue
        d3[k]['scn_to_use'] = scn_to_use
        d3[scn_to_use]['slices_in_this_scn'] += 1
        

    with open(picklename, mode='wb') as pfile:
        pickle.dump(d3,pfile)
        
    with open(csvname, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        #writer=csv.writer(csvfile)
        slicelist = sorted(list(d3.keys()))
        for k in slicelist: writer.writerow(d3[k])

