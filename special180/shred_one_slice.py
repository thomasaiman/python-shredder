# -*- coding: utf-8 -*-


import os
import numpy as np
import openslide
import shutil
import pickle
from scipy import ndimage
from PIL import Image
from shred_utils import (region_info, extractor, bg_color_finder, df_fixer,
                         scn_xcorr, region_selector)

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
    mask_black = np.all(hires_img==0,axis=2)
    mask_black = ndimage.binary_fill_holes(mask_black)
    mask_white = np.all(hires_img==255,axis=2)
    mask_white = ndimage.binary_fill_holes(mask_white)
    mask_pre = np.logical_or(mask_black, mask_white)
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
    
    
    #%% determine which regions are in the slice
    region_selector(si, slidePtr)
    
    
    #this is a good point to save stuff for debugging purposes
    pname=os.path.join(tmp_dir, 'all_si.p')
    with open(pname, mode='wb') as f:
        pickle.dump(si, f)
         
    #%%make a nice blank image 
    #copying this file will be faster than encoding + compressing an array
    blankPath = os.path.join(slice_dir, 'blank.png')
    b = Image.new(mode='RGB', size=(256,256), color=tuple(bg_color))
    b.save(blankPath, optimize=True)
    
    #%% open + fix deformation field
    print('\nfilling deformation field')
    df_R,df_C = df_fixer(df_path)
    slice_info['df_R']=df_R
    slice_info['df_C']=df_C
    
    
    #%% deform level 2 image
    print('\nlev2 deformation')
    #interpolate df values for larger images
    df_R2 = np.asarray(Image.fromarray(df_R).resize((4096,4096), resample=Image.LANCZOS))
    df_C2 = np.asarray(Image.fromarray(df_C).resize((4096,4096), resample=Image.LANCZOS))
    df_R2 = df_R2*16
    df_C2 = df_C2*16
    
    lev2_deformed = np.zeros([4096, 4096, 3],dtype='uint8')
    manual_img=np.asarray(manual_img)
    for ii in [0,1,2]:
        #df values that fall outside the original image are turned to bg_color
        lev2_deformed[:,:,ii] = ndimage.map_coordinates(manual_img[:,:,ii], [df_R2,df_C2], mode='constant', cval=bg_color[ii])
    

    
    #%% shred levels 2,3,4
    #We have a fully deformed image from level 2. This can be downsampled to do
    #levels 3 and 4.
    if type(lev2_deformed) is not Image.Image:
        lev2_deformed=Image.fromarray(lev2_deformed)
        
    for level in (2,3,4):
        print('level {} shred'.format(level))
        nrows = 4**(4-level)
        tile_sz=256 #pixels
        deformed_img = lev2_deformed.resize((nrows*tile_sz, nrows*tile_sz), resample=Image.LANCZOS)
        levelDir = os.path.join(slice_dir, 'level_{}'.format(level))
        os.makedirs(levelDir, exist_ok=True)
        for ii in range(1, nrows+1):
            rowDir = os.path.join(levelDir, 'row_{:0>3}'.format(ii))
            os.makedirs(rowDir, exist_ok=True)
            for jj in range(1, nrows+1):
                r = tile_sz*(ii-1)
                c = tile_sz*(jj-1)
                imgPath = os.path.join(rowDir, 'col_{:0>3}.png'.format(jj))
                deformed_img.crop((c,r,c+tile_sz,r+tile_sz)).save(imgPath)
            

    #%% levels 1 & 0
    """
    The higher resolution levels can't be loaded completely into memory all
    at once. We will extract and deform one tile at a time.
    
    image resize interpolation uses a bicubic method which uses a 4x4 neighborhood
    for each point. In order to make the df interpolation return smooth values
    across tiles, we must have a 2 pixel overlap between each df chunk. The
    chunk is interpolated, and then the extra edge values (8 pixels on each
    side) are cropped off. In order to handle the edge cases, we must pad our
    big df array by 2 in each direction.
    """
    df_R2 = np.asarray(Image.fromarray(df_R).resize((4096,4096), resample=Image.LANCZOS))
    df_C2 = np.asarray(Image.fromarray(df_C).resize((4096,4096), resample=Image.LANCZOS))
    df_R2 = df_R2*16
    df_C2 = df_C2*16
    df_R2 = np.pad(df_R2, [2,2], mode = 'edge')
    df_C2 = np.pad(df_C2, [2,2], mode = 'edge')
    
    for level in (1,0):
        scale_fac = 4**(2-level)
        szR = int(lev0_size[1]/(4**level))
        szC = int(lev0_size[0]/(4**level))
        num_rows = 4**(4-level)
        num_cols = num_rows
        df_chunk_size = int(4096/num_rows)
        sz = (df_chunk_size+4)*scale_fac
        
        levelDir = os.path.join(slice_dir, 'level_{}'.format(level))
        os.makedirs(levelDir, exist_ok=True)
        tile = np.zeros([256, 256, 3], dtype='uint8')
        
        for ii in range(1,num_rows+1):
            print('level {} row {} / {}'.format(level, ii, num_rows))
            rowDir = os.path.join(levelDir, 'row_{:0>3}'.format(ii))
            os.makedirs(rowDir, exist_ok=True)
            r = df_chunk_size*(ii-1)
            
            for jj in range(1,num_cols+1):
                imgPath = os.path.join(rowDir, 'col_{:0>3}.png'.format(jj))
                
                #interpolate/expand a "tile" of the DF
                c = df_chunk_size*(jj-1);
                df_Ra = df_R2[r:r+df_chunk_size+4, c:c+df_chunk_size+4]
                df_Ca = df_C2[r:r+df_chunk_size+4, c:c+df_chunk_size+4]
                sz = (df_chunk_size+4)*scale_fac
                df_Rb = np.asarray(Image.fromarray(df_Ra).resize((sz,sz), resample=Image.BICUBIC))
                df_Cb = np.asarray(Image.fromarray(df_Ca).resize((sz,sz), resample=Image.BICUBIC))
                df_Rb = df_Rb * scale_fac
                df_Cb = df_Cb * scale_fac
                df_Rc = df_Rb[2*scale_fac:-2*scale_fac,2*scale_fac:-2*scale_fac]
                df_Cc = df_Cb[2*scale_fac:-2*scale_fac,2*scale_fac:-2*scale_fac]
                
                #determine which pixels from the region of interest are needed
                #for this tile
                minR = np.floor(df_Rc.min()).astype('int')        
                maxR = np.ceil(df_Rc.max()).astype('int')
                minC = np.floor(df_Cc.min()).astype('int') 
                maxC = np.ceil(df_Cc.max()).astype('int')
                
                #convert these values to their equivalents in the sideways
                #manual collage image.
                c1 = szC - maxC
                c2 = szC - minC
                r1 = szR - maxR
                r2 = szR - minR
                bounds = (r1,r2,c1,c2)
                #get that from the slide and rotate to the correct orientation
                pre_tile = extractor(slidePtr, SR, level, bounds)
                
                if pre_tile is None:
                    shutil.copy(blankPath, imgPath)
                else:
                    #rotate to the correct orientation
                    pre_tile = np.rot90(pre_tile,-2)
    
                    #fix bg prior to deform
                    #regions sometimes contain blank edges, so we must use this approach
                    mask_pre = np.any(pre_tile,axis=2)
                    mask_pre = ndimage.binary_fill_holes(mask_pre)
                    mask_pre = np.logical_not(mask_pre)
                    pre_tile[mask_pre,:] = bg_color
    
                    #correct the deformation coordinates to work within the small chunk we extracted
                    df_Rc = df_Rc - minR
                    df_Cc = df_Cc - minC
                    
                    #do the image deformation and write to disk
                    for ii in [0,1,2]:
                        tile[:,:,ii] = ndimage.map_coordinates(pre_tile[:,:,ii], [df_Rc,df_Cc], mode='constant', cval=bg_color[ii])
                        
                    Image.fromarray(tile).save(imgPath, optimize=True)
    
    
