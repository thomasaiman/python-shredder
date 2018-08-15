# -*- coding: utf-8 -*-

import os
from file_utils import desktop_cleanup, path_checker, scn_solver
from shred_one_slice import shred_one_slice

slice_info=dict()
si = slice_info
#slice_info['project'] = os.environ['PROJECT']
#slice_info['subject'] = os.environ['SUBJECT']
#slice_info['monkey_name'] = os.environ['MONKEY_NAME']
#slice_info['stain'] = os.environ['STAIN']
slice_info['slice_num'] = 473
slice_info['tmp_dir'] = fr"D:\VUIIS\histology_shredder\mj-myelin-slice-{si['slice_num']}"
slice_info['stor_dir'] = r'D:\VUIIS' 
slice_info['pickle_path'] = r"C:\Users\Thomas\Desktop\VUIIS\histology_shredder\slice-relationships\nSMDA-mj-histo-myelin.p"

scn_to_use, slices_in_this_scn = scn_solver(si['pickle_path'], si['slice_num'])
slice_info['scn_to_use'] = scn_to_use
slice_info['slices_in_this_scn'] = slices_in_this_scn

#accre_xnat_downloader(slice_info)

paths = path_checker(si['tmp_dir'])
slice_info.update(paths)

shred_one_slice(slice_info)

desktop_cleanup(si['tmp_dir'], si['stor_dir'])
