# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
from PIL import Image
from upsampled_IDFT import upsampled_IDFT








def df_fixer(df_path):
    """
    There are usually holes/blank spots toward the outside of the df. The
    extrapolation we do isn't totally necessary, but it makes the later
    deformation have cleaner edges and the background color is much easier to fix.
    """
    
    df = np.fromfile(df_path, dtype='float32', count=-1)
    df = df.reshape((256, 256, 3), order='F')
    df_R = df[:, :, 0]
    df_C = df[:, :, 1]
    # df_R += 1
    # df_C += 1
    
    # fill holes in df with spline interpolation/extrapolation
    [r1, c1] = np.where(df_R != -1)
    [rq, cq] = np.where(df_R == -1)
    splineR = SmoothBivariateSpline(r1, c1, df_R[r1, c1])
    df_R[rq, cq] = splineR.ev(rq, cq)
    
    [r1, c1] = np.where(df_C != -1)
    [rq, cq] = np.where(df_C == -1)
    splineC = SmoothBivariateSpline(r1, c1, df_C[r1, c1])
    df_C[rq, cq] = splineC.ev(rq, cq)
    return(df_R,df_C)







def bg_color_finder(im):
    #im should be a PIL.Image object
    h=im.histogram()
    
    #split the list that Image.histogram() returns into color channels
    r=h[0:256]
    g=h[256:512]
    b=h[512:768]
    
    color = []
    #find the most common value for each color channel
    for c in (r,g,b):
        color.append( c.index(max(c)) )
    
    return(color)






def region_info(slidePtr):
    #determine how many regions exist
    #make a list with a dict for each region
    #this assumes that the slide properites are labeled in a certain way
    props = dict(slidePtr.properties)
    data = list()
    for k in props.keys():
        if k[0:17] == 'openslide.region[' and k[18:] == '].height':
            print('Region found: ' + k[17])
            data.append({'name':k[17]})
    
    for k in data:
        k['x0'] = int(props['openslide.region[{}].x'.format(k['name'])])
        k['y0'] = int(props['openslide.region[{}].y'.format(k['name'])])
        k['w0'] = int(props['openslide.region[{}].width'.format(k['name'])])
        k['h0'] = int(props['openslide.region[{}].height'.format(k['name'])])
            
    return(data)







def scn_xcorr(slidePtr, SR, manual_img):
    # calculate where each scn region best matches with the manual image
    # updates the SR list that is passed to it
    
    print('\npreprocessing manual_img')
    # -use the manually extracted manual_img, but downsampled so that it's
    #   possible to do the fft with the RAM of a normal PC
    # -convert img to grayscale.
    # -normalize img so dark and light parts have equal weighting.
    
    gray_img = manual_img.transpose(Image.ROTATE_90).convert(mode='F') #uses perceptual luminance weighted sum to convert RGB to F
    gray_img = np.asarray(gray_img)
    avg_brightness = gray_img.mean()
    manual_norm = gray_img-avg_brightness
    
    xpad=max(SR, key=lambda x:x['w0'])['w0']//16
    xpad=max(xpad, manual_norm.shape[1]) + 1000
    ypad=max(SR, key=lambda x:x['h0'])['h0']//16
    ypad=max(ypad, manual_norm.shape[0]) + 1000
    manual_pad = np.zeros((ypad,xpad), dtype='float32')
    manual_pad[0:gray_img.shape[0], 0:gray_img.shape[1]] = manual_norm
    manual_ft = np.fft.fft2(manual_pad)
    
    #calculate the cross-correlation of the Match img with the slide
    #we are using the FFT method because it is much faster
    print('Calculating cross correlations')
    for v in SR:
        '''    
        -extract the region from lvl 1 of the slide 
        
        -we do the cross-correlation with the lev 2 resolution to avoid
        large ffts, but it's much more accurate with a downsized+blurred
        version of lev 1. This is because openslide-matlab doesn't blur for
        higher levels, it just downsamples. This downsampling doesn't always match
        up well with the manual collage
        '''
        sz0 = np.array([v['w0'],v['h0']])
        sz1 = (sz0/4).round().astype(int)
        region_lev1 = slidePtr.read_region((v['x0'],v['y0']),1,tuple(sz1))
        sz2 = (sz0/16).round().astype(int)
        region_lev2 = region_lev1.resize(tuple(sz2), resample=Image.LANCZOS)
        
        gray_img = region_lev2.convert(mode='F') #uses perceptual luminance weighted sum to convert RGB to F
        gray_img = np.asarray(gray_img)
        avg_brightness = gray_img.mean()
        region_norm = gray_img-avg_brightness
        rset = (region_norm**2).sum()
        region_pad = np.zeros(manual_pad.shape, dtype='float32')
        #the template has to mirrored for cross-correlation
        region_pad[gray_img.shape[0]:0:-1, gray_img.shape[1]:0:-1] = region_norm
        region_ft = np.fft.fft2(region_pad)
        
        #local f rses
        man2 = manual_pad**2
        kernel = region_pad!=0
        man2f = np.fft.fft2(man2)
        kernelf = np.fft.fft2(kernel)
        local_f_rses = np.fft.ifft2(man2f*kernelf)
        local_f_rses = np.real(local_f_rses)
        
        #phase correlation
        xcorr_ft = manual_ft*region_ft;
        cor_match = np.real(np.fft.ifft2(xcorr_ft))
        cor_match = cor_match/local_f_rses
        maxR,maxC = np.unravel_index(cor_match.argmax(), cor_match.shape)
        mymax = cor_match.argmax()
        '''
        #Subpixel registration - compute a more detailed cross-correlation, but only in a small
        #area around the peak.
        [cor_match2,rows,cols] = upsampled_IDFT(xcorr_ft,16,maxR,maxC)
        cor_match2 = cor_match2.real
        maxR2, maxC2 = np.unravel_index(cor_match2.argmax(), cor_match2.shape)
        mymax = cor_match2[maxR2,maxC2]
        maxR = rows[maxR2]
        maxC = cols[maxC2]
        '''
        
        #Use (xcorr peak value)/(area of region) as a measure of
        #whether the region is part of the slide.
        area = v['w0']*v['h0']
        v['norm_corr'] = mymax/rset
        
        #convert the cross-correlation max location to the value we need to shift the 
        #small region to accurately place it on the lev 0 manual_img
        v['lm_R0'] = int(maxR*16 - v['h0'])
        v['lm_C0'] = int(maxC*16 - v['w0'])
        print(v)







def extractor(slidePtr,SR,level,bounds):
    """
    'lm' stands for 'light space (manual)'
    'lm_*' variables are coordinates on the manual collage image
    'ls' stands for 'light space (slide)'
    'ls_*' variables are coordinates on the whole slide image
    
    
    Inputs:
    slidePtr - pointer object created by openslide.OpenSlide()
    SR - struct with fields x0, y0, w0, h0, shiftR0, shiftC0. x0, y0, w0, h0
        tell us where the region is on the slide. shiftR0 and shiftC0 tell us 
        where the region is on the manual collage image. SR should have only 
        the regions that are relevant to our slice.
    level - number corresponding to the openslide object level we will extract
        from
    bounds - tuple of (rmin, rmax, cmin, cmax) of the area we want in lm space

    
    Outputs:
    lm_chunk - an MxNx3 RGB array of the lm space image data in the requested 
        region. This can also be None if there is no image data in the 
        requested region.
    
    """  
    lvl_scale = int(4**level)
    bounds = np.array(bounds).astype(int)
    rmin,rmax,cmin,cmax = bounds
    bounds0 = np.array(bounds).astype(int)*lvl_scale
    for s in SR:
        #determine which parts of each region are present in the desired lm chunk
        s['lm_Cmin'] = max(bounds0[2], s['lm_C0']         )
        s['lm_Cmax'] = min(bounds0[3], s['lm_C0']+s['w0'] )
        s['lm_Rmin'] = max(bounds0[0], s['lm_R0']         )
        s['lm_Rmax'] = min(bounds0[1], s['lm_R0']+s['h0'] )
                
        #if the section is part of our chunk
        if s['lm_Cmin'] <= s['lm_Cmax'] and s['lm_Rmin'] <= s['lm_Rmax']:

            s['in_chunk'] = True
        else:
            #tell this function to ignore this region
            s['in_chunk'] = False   
    
    
    #if there's nothing of interest send a signal to just copy the existing blank file
    if not any([s['in_chunk'] for s in SR]):
        return(None)
        
    
    #take the appropriate part of the slide chunk and put it in the manual
    #collage chunk in the correct place
    lm_chunk = np.zeros([rmax-rmin+1, cmax-cmin+1, 3], dtype='uint8') 
    for s in SR:
        #if the region is part of our chunk
        if s['in_chunk']:
            #determine where those parts exist in the whole slide image
            s['ls_xmin'] = s['lm_Cmin'] - s['lm_C0'] + s['x0']
            s['ls_xmax'] = s['lm_Cmax'] - s['lm_C0'] + s['x0']
            s['ls_ymin'] = s['lm_Rmin'] - s['lm_R0'] + s['y0']
            s['ls_ymax'] = s['lm_Rmax'] - s['lm_R0'] + s['y0']
            ls_w = s['ls_xmax'] - s['ls_xmin']
            ls_h = s['ls_ymax'] - s['ls_ymin']
            
            #read that part of the slide
            ls_w = ls_w//lvl_scale + 1
            ls_h = ls_h//lvl_scale + 1
            ls_chunk = slidePtr.read_region((s['ls_xmin'], s['ls_ymin']), level, (ls_w, ls_h))
            ls_chunk = np.asarray(ls_chunk.convert(mode='RGB'))
            
            #place the ls_chunk in the lm_chunk
            a = np.array([s['lm_Rmin'], s['lm_Cmin']])
            a = a//lvl_scale - [rmin, cmin]
            b = a + [ls_h, ls_w]
            lm_chunk[a[0]:b[0], a[1]:b[1], :] = ls_chunk
            
    return(lm_chunk)        
        
        
