# -*- coding: utf-8 -*-

'''

This function computes a small, high detail section of an IDFT with low
memory requirements.  It does this with matrix multiplication. This is
preferable to bilinear or bicubic interpolation of the lower resolution
IDFT because this function uses the all of the data in the 'input' matrix
to upsample. Useful for image template matching.

Strongly influenced by  Manuel Guizar's 'dftregistration.m' and this paper:
Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
"Efficient subpixel image registration algorithms," Opt. Lett. 33, 
156-158 (2008).

Inputs: 
input - the 2D matrix you want to perform a IDFT on
usfac - upsampling factor 
rowin - a numpy array [min max] of rows you want upsampled. If only a single
        value is provided, it will default to [rowin-2, rowin+2]
colin - like 'rowin' but for columns

Outputs: 
output - the 2D upsampled IDFT within the specified region 
rows - a vector specifying where each row in the upsampled IDFT would 
       correspond to in the normal IDFT 
cols - like 'rows' but for the columns
'''

import numpy as np

def upsampled_IDFT(inputft,usfac, rowin, colin):
    
    if rowin.size == 1:
        rowin = np.array([rowin-2, rowin+2])
    if colin.size == 1:
        colin = np.array([colin-2, colin+2])
    
    
    #make a DFT matrix that operates on cols
    #pick the rows of the IDFT that we want to calculate
    rows = np.linspace(rowin[0], rowin[1], usfac*rowin.ptp()+1)
    M = inputft.shape[0]
    fc = np.arange(0,M,1)
    '''
        %when upsampling a IDFT we have to actually convert the upper half of
        %frequencies to their negative equivalents so that the interpolated
        %points fall where we expect them to.
        %As an example, do this and look at what happens:
        %{
        w = 0:10;
        a1 = exp(2*pi*1i*7*w/10);
        b1 = exp(2*pi*1i*-3*w/10);
        figure;
        subplot(2,2,1); plot(w,real(a1),w,imag(a1)); title('a1')
        subplot(2,2,2); plot(w,real(b1),w,imag(b1)); title('b1')
        w = 0:0.1:10;
        a2 = exp(2*pi*1i*7*w/10);
        b2 = exp(2*pi*1i*-3*w/10);
        subplot(2,2,3); plot(w,real(a2),w,imag(a2)); title('a2')
        subplot(2,2,4); plot(w,real(b2),w,imag(b2)); title('b2');
        %}
    '''
    fc[fc>M/2]=fc[fc>M/2]-M
    w = 2*np.pi*1j*1/M
    a1,a2 = np.meshgrid(fc, rows);
    a = a1*a2;
    Wc = np.exp(a*w)/M
    
    # make a IDFT matrix that operates on rows
    #pick the columns of the IDFT that we want to calculate    
    cols = np.linspace(colin[0], colin[1], usfac*colin.ptp()+1)
    N = inputft.shape[1]
    fr = np.arange(0,N,1)
    fr[fr>N/2]=fr[fr>N/2]-N
    w = 2*np.pi*1j*1/N
    a1,a2 = np.meshgrid(cols, fr);
    a = a1*a2;
    Wr = np.exp(a*w)/N
    
    output = Wc @ inputft @ Wr
    return (output,rows,cols)



