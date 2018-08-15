# -*- coding: utf-8 -*-
import os

#make sure there is exactly one .scn, .tif, and .df file for the slice
def path_checker(tmp_dir):
    paths = dict()
    counts = {'scn':0, 'tif':0, 'df':0}
    for file in os.listdir(tmp_dir):
        if file.endswith(".scn"):
            paths['scn_path'] = os.path.join(tmp_dir,file)
            counts['scn'] += 1
        elif file.endswith(".tif"):
            paths['man_img_path'] = os.path.join(tmp_dir,file)
            counts['tif'] += 1
        elif file.endswith(".df"):
            paths['df_path'] = os.path.join(tmp_dir,file)
            counts['df'] += 1
    
    
    
    
    if all(v==1 for v in counts.values()):
        print(paths)
        print('File paths look good \n')
        return(paths)
    else:
        raise ValueError('Wrong number of files in {}: {}'.format(tmp_dir, counts))
    
    

