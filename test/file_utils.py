# -*- coding: utf-8 -*-
import os
import shutil
import pickle



def scn_solver(pickle_path, slice_num):
    #d is a dict with a key for each slice number. That slice's entry is a dict
    #with fieldnames =  ('slice_num', 'df_files', 'scn_files', 'hi_res_files',
    #                   'scn_to_use', 'slices_in_this_scn')
    
    with open(pickle_path, mode='rb') as f: 
        d=pickle.load(f)
    if d[slice_num]['df_files'] == 0:
        raise ValueError("Slice {} doesn't have a df file".format(slice_num))
    if d[slice_num]['hi_res_files'] == 0:
        raise ValueError("Slice {} doesn't have a hi res tiff file".format(slice_num))
    scn_to_use = int(d[slice_num]['scn_to_use'])
    slices_in_this_scn = d[scn_to_use]['slices_in_this_scn']
    return(scn_to_use, slices_in_this_scn)





def path_checker(tmp_dir):
    #make sure there is exactly one .scn, .tif, and .df file for the slice
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
    for k,v in paths.items(): print(k,':',v)
    if all(v==1 for v in counts.values()):
        print('File paths look good')
        return(paths)
    else:
        raise ValueError('Wrong number of files in {}: {}'.format(tmp_dir, counts))






def accre_xnat_downloader(slice_info):
    si=slice_info
    os.makedirs(si['tmp_dir'], exist_ok=True)
    session = '-'.join( (si['project'], si['monkey_name'], 'histo', si['stain']) )
    scan_hi_res='-'.join( (si['monkey_name'], 'histo', si['stain'], 'hi_res_files') )
    scan_resource='slice_{}'.format(si['slice_num'])
    assessor='-'.join( (si['monkey_name'], 'histo', si['stain'], 'df_files') )
    asses_resource='slice_{}-light_block3_01'.format(si['slice_num'])
    #get df and hi res tif files
    cmd1 = 'source ~/xnatenv/bin/activate'
    cmd2 = 'source ~/.xnat_profile'
    cmd3 = ' '.join((
                    'Xnatdownload',
                    '--project={}'.format(si['project']),
                    '--subj={}'.format(si['subject']),
                    '--sess={}'.format(session),
                    '--scantype={}'.format(scan_hi_res),
                    '--rs={}'.format(scan_resource),
                    '--assessortype={}'.format(assessor),
                    '--ra={}'.format(asses_resource),
                    '--directory={}'.format(si['tmp_dir']),
                    '--oneDirectory'    ))
    cmd4 = 'source deactivate'
    
    cmd = ' && '.join((cmd1, cmd2, cmd3, cmd4))
    print('command is: \n', cmd, '\n')
    os.system(cmd)     
    #get scn file
    scan_scn = '-'.join( (si['monkey_name'], 'histo', si['stain'], 'scn_files') )
    scan_resource='slice_{}'.format(si['scn_to_use'])
    cmd3 = ' '.join((
                'Xnatdownload',
                '--project={}'.format(si['project']),
                '--subj={}'.format(si['subject']),
                '--sess={}'.format(session),
                '--scantype={}'.format(scan_scn),
                '--rs={}'.format(scan_resource),
                '--directory={}'.format(si['tmp_dir']),
                '--oneDirectory'    ))
    cmd = ' && '.join((cmd1, cmd2, cmd3, cmd4))
    print('command is: \n', cmd, '\n')
    os.system(cmd)
    


def accre_cleanup(tmp_dir, stor_dir):
    os.makedirs(stor_dir, exist_ok=True)
    os.chdir(tmp_dir)
    tar_name= os.path.split(tmp_dir)[1]
    shutil.make_archive(base_name=tar_name, format='gztar',base_dir=os.path.split(tmp_dir)[1])
    shutil.move(src=tar_name+'.tar.gz', dst=stor_dir)
    os.chdir(stor_dir)
    shutil.rmtree(tmp_dir)
    
    
    
def desktop_cleanup(tmp_dir, stor_dir):
    os.makedirs(stor_dir, exist_ok=True)
    os.chdir(tmp_dir)
    tar_name= os.path.split(tmp_dir)[1]
    shutil.make_archive(base_name=tar_name, format='gztar',base_dir=os.path.split(tmp_dir)[1])
    shutil.move(src=tar_name+'.tar.gz', dst=stor_dir)
    os.chdir(stor_dir)
    
    
