# -*- coding: utf-8 -*-

from file_utils import desktop_cleanup, path_checker, scn_solver
from shred_one_slice import shred_one_slice

slice_info=dict()
si = slice_info
slice_info['project'] = 'nSMDA'
slice_info['subject'] = 'new_mj'
slice_info['monkey_name'] = 'mj'
slice_info['stain'] = 'myelin'
slice_info['slice_num'] = 628

slice_info['tmp_dir'] = fr"D:\VUIIS\histology_shredder\{si['monkey_name']}-{si['stain']}-slice-{si['slice_num']}"
slice_info['stor_dir'] = r'D:\VUIIS' 
slice_info['pickle_path'] = fr"C:\Users\Thomas\Desktop\VUIIS\histology_shredder\slice-relationships\{si['project']}-{si['monkey_name']}-histo-{si['stain']}.p"

scn_to_use, slices_in_this_scn = scn_solver(si['pickle_path'], si['slice_num'])
slice_info['scn_to_use'] = scn_to_use
slice_info['slices_in_this_scn'] = slices_in_this_scn

#accre_xnat_downloader(slice_info)

paths = path_checker(si['tmp_dir'])
slice_info.update(paths)

shred_one_slice(slice_info)

#desktop_cleanup(si['tmp_dir'], si['stor_dir'])
print('Done!')