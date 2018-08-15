# -*- coding: utf-8 -*-


import os
import numpy as np
import openslide
import shutil
import pickle
from scipy import ndimage
from PIL import Image
from shred_utils import region_info, extractor, bg_color_finder, df_fixer, scn_xcorr

def shred_one_slice(slice_info):
    si = slice_info
    
    tmp_dir = si['tmp_dir']
    slice_dir = os.path.join(tmp_dir, os.path.split(tmp_dir)[1])
    os.makedirs(slice_dir, exist_ok=True)
    
    man_img_path = si['man_img_path']
    scn_path = si['scn_path']
    df_path = si['df_path']
    slidePtr = openslide.OpenSlide(scn_path)
    
    #%% get a list of slide regions and their properties
    print()
    SR = region_info(slidePtr)
    si['SR'] = SR
    
    
    
    #%% open the manual image, fill background and downsample it to lev 2
    #Pillow might throw a DecompressionBombError if it thinks the image is too big
    #so we're going to disable that check
    print('\nreading manual_img')
    Image.MAX_IMAGE_PIXELS = None
    hires_img = Image.open(man_img_path).convert(mode='RGB')
    lev0_size = np.array(hires_img.size)*4
    lev2_size = (lev0_size/16).round().astype(int)
    print('fixing bg holes')
    bg_color=bg_color_finder(hires_img)
    si['bg_color']=bg_color
    print('bg_color is: {}'.format(bg_color))
    hires_img=np.asarray(hires_img)
    mask_pre = np.any(hires_img,axis=2)
    mask_pre = ndimage.binary_fill_holes(mask_pre)
    mask_pre = np.logical_not(mask_pre)
    hires_img.flags['WRITEABLE']=True
    hires_img[mask_pre,:] = bg_color
    print('resizing manual_img to lv2')
    hires_img = Image.fromarray(hires_img)
    manual_img = hires_img.resize(tuple(lev2_size), resample=Image.LANCZOS)
    hires_img.close()
    si['manual_img']=manual_img
    si['lev0_size']=lev0_size
    #%% calculate where each scn region best matches with the manual image
    scn_xcorr(slidePtr, SR, manual_img)
    
    #this is a good point to save stuff for debugging purposes
    pname=os.path.join(tmp_dir, 'all_si.p')
    with open(pname, mode='wb') as f:
        pickle.dump(si, f)
        
    #%% determine which regions are in the slice
    """
    -each scn may contain any number of slices
    -each scn may contain any number of regions
    -assume that there are an equal number of regions for each slice
    -the slice's regions should be those with the highest cross-correlation value
    """
    SR=slice_info['SR']
    SR.sort(key=lambda x:x['norm_corr'], reverse=True)
    num_to_keep = round(len(SR)/si['slices_in_this_scn'])
    oldSR = SR
    SR = SR[0:num_to_keep]
    print('regions kept:', [x['name'] for x in SR])
    
    #check if any of the regions has moved way more than the others
    #this would indicate that an extra, unwanted region has been automatically included
    print('\nchecking for unwanted regions:')
    a = [v['x0']-v['lm_C0'] for v in SR]
    b = [v['y0']-v['lm_R0'] for v in SR]
    ranges = np.array([b,a]).ptp(axis=1)
    print('shift ranges:', ranges)
    print('acceptable shift ranges:', lev0_size*0.15)
    if any(ranges>lev0_size*0.15):
        #save a copy of each region (for human review) and raise an error
        imgPath = os.path.join(slice_dir, 'slide.png') 
        slidePtr.get_thumbnail((2000,2000)).save(imgPath)
        imgPath = os.path.join(slice_dir, 'man_img.png') 
        manual_img.save(imgPath)
        for v in oldSR:
            sz0 = np.array([v['w0'],v['h0']])
            sz2 = (sz0/16).round().astype(int)
            region_lev3 = slidePtr.read_region((v['x0'],v['y0']),2,tuple(sz2))
            imgPath = os.path.join(slice_dir, '{}.png'.format(v['name']) )
            region_lev3.save(imgPath)
        store_path = os.path.join(slice_info['stor_dir'], os.path.split(slice_dir)[1])
        #shutil.move(src=tmp_dir, dst=store_path)
        raise ValueError('The automatic region choices appear to be wrong')
    
    
    
    
        
   