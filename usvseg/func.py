import math
import numpy as np
import scipy.signal
import scipy.io
import copy
import cv2
import os
import wave
import csv
from tqdm import tqdm
import glob
import scipy.io.wavfile

# ---- Memo: difference between matlab and python ----
# min(), max() outputs when the inputs include nan
# processing axis of fft() when the input is a matrix
# ----------------------------------------------------

default_tapers = scipy.signal.windows.dpss(512,3,6)
default_tapers = default_tapers.T

class FigData:

    def __init__(self, x_disp, y_disp, mtsp_disp, fltnd_disp, 
                 nonsylzone, margzone, tvec, fvec, freqtrace, fs, freqmin, freqmax, info_str):
        self.x_disp = copy.deepcopy(x_disp)
        self.y_disp = copy.deepcopy(y_disp)
        self.mtsp_disp = copy.deepcopy(mtsp_disp)
        self.fltnd_disp = copy.deepcopy(fltnd_disp)
        self.nonsylzone = copy.deepcopy(nonsylzone)
        self.margzone = copy.deepcopy(margzone)
        self.tvec = copy.deepcopy(tvec)
        self.fvec = copy.deepcopy(fvec)
        self.freqtrace = copy.deepcopy(freqtrace)
        self.fs = copy.deepcopy(fs)
        self.freqmin = copy.deepcopy(freqmin)
        self.freqmax = copy.deepcopy(freqmax)
        self.info_str = copy.deepcopy(info_str)

def multitaperspec(wav,fs,fftsize,timestep,tapers=None):

    if tapers is None:
        tapers = default_tapers

    step = round(timestep*fs)
    wavlen = wav.shape[0]
    ntapers = tapers.shape[1]
    nsteps = math.floor((wavlen-fftsize+step)/step)
    spgsize = fftsize/2+1

    n = wav.strides[0]
    wavslice = np.lib.stride_tricks.as_strided(wav, shape=(nsteps, fftsize), strides=(n*step,n))
    wavslice = wavslice.T

    spmat = np.zeros([int(spgsize),nsteps,ntapers])

    for n in range(ntapers):
        a = np.tile(tapers[:,n], [nsteps,1]).T

        ww = wavslice * a

        ft = np.fft.rfft(ww.T, fftsize)
    
        ft = ft.T
        spmat[:,:,n] = np.abs(ft[np.arange(int(fftsize/2+1)),:])

    """
    ## joblib ver, but not fast

    import joblib

    def parjob(n, tapers, nsteps, wavslice, fftsize):
        a = np.tile(tapers[:,n], [nsteps,1]).T
        ww = wavslice * a
        ft = np.fft.rfft(ww.T, fftsize)
        ft = ft.T
        return np.abs(ft[np.arange(int(fftsize/2+1)),:])
    
    result = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(parjob)(n, tapers, nsteps, wavslice, fftsize) for n in range(ntapers))
    
    for n in range(ntapers):
        spmat[:,:,n] = result[n]
    """
        
    mtsp = 20*np.log10(np.mean(spmat,axis = 2)*math.sqrt(1/(2*math.pi*fftsize)))

    return mtsp

def flattening(mtsp, med=None, liftercutoff=3):
    
    fftsize = (mtsp.shape[0]-1)*2

    cep = np.fft.fft(np.concatenate([mtsp, np.flipud(mtsp[1:-1,:])]), axis=0)
    lifter = np.ones(cep.shape)
    lifter[0:liftercutoff,:] = 0
    lifter[-liftercutoff:,:] = 0

    temp = np.real(np.fft.ifft(cep*lifter, axis=0))
    liftered = temp[0:int(fftsize/2+1),:]

    if med is None:
        med = np.median(liftered,axis=1)
    
    liftmed = liftered - np.tile(med, [liftered.shape[1],1]).T

    fltnd = scipy.ndimage.median_filter(liftmed, [5, 1])
    
    return fltnd, med

def estimatethresh(fltnd,fs,freqmin,freqmax,threshval):
    
    fftsize = (fltnd.shape[0]-1)*2
    fnmin = math.floor(freqmin/fs*fftsize)
    fnmax = math.ceil(freqmax/fs*fftsize)
    cut = fltnd[fnmin:fnmax+1,:]
    bin = np.arange(-0.05, 10, 0.1)
    bc = bin[0:-1] + np.diff(bin)/2
    h, _ = np.histogram(cut, bins=bin)
    I = np.argwhere(h<h[0]/2) 
    fwhm = bc[np.min(I)-1]*2
    sigma = fwhm/2.35
    thresh = sigma*threshval

    return thresh

def thresholding(fltnd,fs,freqmin,freqmax,thresh):
        
    fftsize = (fltnd.shape[0]-1)*2
    fnmin = math.floor(freqmin/fs*fftsize)
    fnmax = math.ceil(freqmax/fs*fftsize)
    mask = np.zeros(fltnd.shape)
    mask[fnmin:fnmax,:] = 1
    thrshd = (fltnd>thresh)*mask

    return thrshd

def detectonoffset(thrshd,fs,timestep,gapmin,durmin,durmax,margin,onoffthresh=5):

    fftsize = (thrshd.shape[0]-1)*2
    step = round(timestep*fs)
    # onset/offset detection
    onoff = np.max(scipy.signal.lfilter(np.ones(onoffthresh), 1, thrshd, axis=0), axis=0) >= onoffthresh
    # merge fragmented pieces
    ndurmin = round(durmin*fs/step)
    f = scipy.signal.lfilter(np.ones(ndurmin)/ndurmin, 1, np.concatenate([onoff, np.zeros(round(ndurmin/2))]))
    monoff = f[round(ndurmin/2):]>0.5
    monoff[0] = 0
    monoff[-1] = onoff[-1]
    monoff = monoff.astype(int)
    onidx = np.argwhere(np.diff(monoff)>0)+1
    offidx = np.argwhere(np.diff(monoff)<0)+1

    if onidx.shape[0] == 0 or offidx.shape[0] == 0:
        onoffset = np.zeros([0,2])
        onoffsetm = np.zeros([0,2])
        onoffsig = np.zeros([thrshd.shape[1],1])
        contflg = 0
        return onoffset, onoffsetm, onoffsig, contflg

    offidx[-1] = min(offidx[-1], monoff.shape[0]-1)
    if onidx.shape[0] > 0 and monoff[-1] == 1:
        onidx = onidx[:-1]

    # continuity flag: check if the end of read file is "ON"
    contflg = monoff[-1]

    # gap thresholding
    gap = (onidx[1:]-offidx[:-1])*timestep
    gap = gap.ravel() 
    gap = np.concatenate([gap, np.array([0])], axis=0)
    gid = np.argwhere(gap>=gapmin) 
    gid = gid.ravel()
    if gid.shape[0] > 0:
        onidx = np.concatenate([onidx[0, np.newaxis], onidx[gid+1,:]])
        offidx = np.concatenate([offidx[gid,:], offidx[-1, np.newaxis]])
    else:
        onidx = onidx[0, np.newaxis]
        offidx = offidx[-1, np.newaxis]

    # syllable duration threholding
    dur = (offidx-onidx)/fs*step 
    did = np.argwhere(np.logical_and(durmin<=dur, dur<=durmax))
    did = did[:,0]
    onidx = onidx[did]
    offidx = offidx[did]
    tvec = (np.arange(0, thrshd.shape[1])*step+fftsize/2)/fs
    onset = tvec[onidx]
    offset = tvec[offidx]

    if onset.shape[0] == 0 or offset.shape[0] == 0:
        onoffset = np.zeros([0,2])
        onoffsetm = np.zeros([0,2])
        onoffsig = np.zeros([thrshd.shape[1],1])
        contflg = 0
        return onoffset, onoffsetm, onoffsig, contflg

    # margin addition
    onsetm = onset-margin
    offsetm = offset+margin

    # syllables whose margins are overlapped are integrated in onoffsetm but not in onoffset
    idx = np.argwhere((onsetm[1:] - offsetm[:-1])>0)
    idx = idx[:,0]
    onsetI = np.concatenate([onset[0, np.newaxis], onset[idx+1,:]])
    offsetI = np.concatenate([offset[idx,:], offset[-1, np.newaxis]])
    onsetm = onsetI-margin
    onsetm[0] = max(1/fs*step, onsetm[0])
    offsetm = offsetI+margin
    offsetm[-1] = min(thrshd.shape[1]*step/fs,offsetm[-1])
    
    # output
    onoffset = np.concatenate([onset, offset], axis=1)
    onoffsetm = np.concatenate([onsetm, offsetm], axis=1)

    # on/off signal
    temp = np.zeros(onoff.shape)
    onidx2 = np.round((onset*fs-fftsize/2)/step)
    offidx2 = np.round((offset*fs-fftsize/2)/step)
    temp[onidx2.astype(int)] = 1
    temp[offidx2.astype(int)+1] = -1
    onoffsig = np.cumsum(temp)
    
    return onoffset, onoffsetm, onoffsig, contflg

def spectralsaliency(fltnd, tapers=None):

    if tapers is None:
        tapers = default_tapers

    fftsize = (fltnd.shape[0]-1)*2
    tfsp = np.fft.fftshift(np.sum(np.abs(np.fft.fft(tapers,fftsize,axis=0)), axis=1))
    dtfsp = -np.diff(tfsp, 2) # second-order differential
    rng = fftsize/2+ np.arange(-7,6,1)
    rdtfsp = dtfsp[rng.astype(int)]
    salfilt = (rdtfsp-np.mean(rdtfsp))/np.std(rdtfsp)
    fil = scipy.signal.lfilter(salfilt, 1, np.concatenate([fltnd, np.zeros([6, fltnd.shape[1]])], axis=0), axis=0)
    spcsal = fil[6:,:]

    return spcsal

def searchpeak(specsaliency,specamp,ncandidates,bandwidth):
    
    num_steps = specsaliency.shape[1]
    search_range = bandwidth-1
    remove_range = bandwidth*2-1
    peakfreq = np.zeros([num_steps,ncandidates])
    peakfreq[:,:] = np.nan
    peakamp = np.zeros([num_steps,ncandidates])
    peakamp[:,:] = np.nan
    specsaliency[specsaliency<0] = 0

    for n in range(num_steps):
        slice = copy.deepcopy(specsaliency[:,n])
        for c in range(ncandidates):
            mi = np.argmax(slice)
            # center of gravity
            rng = np.arange(max(mi-search_range,0), min(mi+search_range+1, slice.shape[0]), 1, dtype=int)
            temp = copy.deepcopy(specsaliency[rng,n])
            peak = np.sum(temp*rng)/np.sum(temp)
            if np.isnan(peak):
                continue
            # store
            peakfreq[n,c] = peak
            peakamp[n,c] = specamp[mi,n]
            # remove
            idx = np.arange(max(round(peak)-remove_range,0), min(round(peak)+remove_range+1,slice.shape[0]), 1, dtype=int)
            slice[idx] = -np.inf

    return peakfreq,peakamp

def segregatepeak(peakfreq,peakamp,conthr,ampthr):
    peakfreq[peakamp<ampthr] = np.nan
    peakamp[peakamp<ampthr] = np.nan
    # object segregatin with putting object number
    # allow skipping two frames (steps)
    distthr = 0.05 # 5 percent: fixed parameter
    nstep,ncand = peakfreq.shape
    objmat = np.reshape(np.arange(0, nstep*ncand, 1, dtype=float),[nstep, ncand])
    objmat[np.isnan(peakfreq)] = np.nan
    nskip = 2 # can skip max 2 frames if intermediate framse are NaN
    distmat = np.zeros([nstep-3,ncand,nskip+1])
    distmat[:,:,:] = np.nan
    pathmat = np.zeros([nstep-3,ncand,nskip+1])
    pathmat[:,:,:] = np.nan
    for n in range(nstep-nskip-1):
        for m in range(nskip+1):
            temp = np.abs(np.matmul(np.reshape(1/peakfreq[n,:], [-1, 1]),
                                    np.reshape(peakfreq[n+m+1,:], [1, -1]))-1)
            temp[np.isnan(temp)] = np.inf
            mv = np.min(temp, axis=1)
            mid = np.argmin(temp, axis=1) 
            distmat[n,:,m] = mv
            pathmat[n,:,m] = mid

        pm = copy.deepcopy(pathmat)
        pm[distmat>distthr] = np.nan
        pm[np.isnan(distmat)] = np.nan
        
        pp = pm[:,:,0]
        if np.sum(np.logical_not(np.isnan(pm[n,:,0]))) > 0:
            pp = pm[:,:,0]
            x = n+1
        elif np.sum(np.logical_not(np.isnan(pm[n,:,1]))) > 0:
            pp = pm[:,:,1]
            x = n+2
        elif np.sum(np.logical_not(np.isnan(pm[n,:,2]))) > 0:
            pp = pm[:,:,2]
            x = n+3
        
        for m in range(ncand):
            if np.isnan(pp[n,m]) == False:
                if objmat[x,int(pp[n,m])] < objmat[n,m]:
                    val = objmat[x,int(pp[n,m])]
                    objmat[objmat==val] = objmat[n,m]

                else:
                    objmat[x,int(pp[n,m])] = objmat[n,m]

    # thresholding
    objnum = np.unique(objmat)
    objnum = objnum[np.logical_not(np.isnan(objnum))]
    peaks2 = copy.deepcopy(peakfreq)
    ampmat2 = copy.deepcopy(peakamp)
    objmat2 = copy.deepcopy(objmat)
    objlen = np.zeros([objnum.shape[0]])
    objamp = np.zeros([objnum.shape[0]])
    for n in range(objnum.shape[0]):
        idx = np.argwhere(objmat==objnum[n])
        objlen[n] = idx.shape[0]
        objamp[n] = np.mean(ampmat2[objmat==objnum[n]])
    for n in range(objlen.shape[0]):
        if objlen[n] < conthr:
            objlen[n] = np.nan
            peaks2[objmat==objnum[n]] = np.nan
            objmat2[objmat==objnum[n]] = np.nan
            ampmat2[objmat==objnum[n]] = np.nan

    objnum = objnum[np.logical_not(np.isnan(objlen))]
    objamp = objamp[np.logical_not(np.isnan(objlen))]
    objlen = objlen[np.logical_not(np.isnan(objlen))]
    
    # sorting
    peakfreqsg = np.zeros(peaks2.shape)
    peakfreqsg[:,:] = np.nan
    peakampsg = np.zeros(peakamp.shape)
    peakampsg[:,:] = np.nan

    for n in range(nstep):
        on = objmat2[n,:]
        oa = np.zeros(on.shape[0])
        oa[:] = np.nan
        for m in range(oa.shape[0]):
            if np.sum(objnum==on[m]) > 0:
                oa[m] = objamp[objnum==on[m]]

        oa2 = copy.deepcopy(oa)
        oa2[np.isnan(oa)] = -np.inf
        sid = np.argsort(oa2)[::-1]
        peakfreqsg[n,:] = peaks2[n,sid]
        peakampsg[n,:] = ampmat2[n,sid]

    return peakfreqsg, peakampsg

def specpeaktracking(mtsp,fltnd,fs,timestep,freqmin,freqmax,onoffset,margin, bandwidth=9):
        
    if onoffset.shape[0] == 0:
        freqtrace = np.zeros([mtsp.shape[1],4])
        freqtrace[:,:] = np.nan
        amptrace = np.zeros([mtsp.shape[1],4])
        freqtrace[:,:] = np.nan
        maxampval = np.zeros([0])
        maxampidx = np.zeros([0])
        maxfreq = np.zeros([0])
        meanfreq = np.zeros([0])
        cvfreq = np.zeros([0])

        return freqtrace, amptrace, maxampval, maxampidx, maxfreq, meanfreq, cvfreq
    
    fftsize = (fltnd.shape[0]-1)*2
    step = round(timestep*fs)

    # spectral peak saliency
    spcsal = spectralsaliency(fltnd)

    # get peak and reorganize
    ## bandwidth = 9 ; this parameter is now customizable
    ncandidates = 4
    contmin = 10
    ampthr = 0
    nstep = fltnd.shape[1]
    fnmin = math.floor(freqmin/fs*fftsize)
    fnmax = math.ceil(freqmax/fs*fftsize)
    onidx = np.round((onoffset[:,0]-margin)*fs/step)-1 # add margin
    offidx = np.round((onoffset[:,1]+margin)*fs/step)-1 # add margin
    onidx[0] = max(1,onidx[0])
    offidx[-1] = min(nstep-1, offidx[-1])
    
    freqmat = np.zeros([nstep,ncandidates])
    freqmat[:,:] = np.nan
    ampmat = np.zeros([nstep,ncandidates])
    ampmat[:,:] = np.nan
    maxampval = np.zeros(onoffset.shape[0])
    maxampval[:] = np.nan
    maxampidx = np.zeros(onoffset.shape[0], dtype=int)
    maxampfreq = np.zeros(onoffset.shape[0])
    maxampfreq[:] = np.nan
    meanfreq = np.zeros(onoffset.shape[0])
    meanfreq[:] = np.nan
    cvfreq = np.zeros(onoffset.shape[0])
    cvfreq[:] = np.nan

    for n in range(onoffset.shape[0]):
        idx = np.arange(onidx[n], offidx[n]+1, 1, dtype=int)
        peakfreq, peakamp = searchpeak(spcsal[fnmin:fnmax+1,idx],fltnd[fnmin:fnmax+1,idx],ncandidates,bandwidth)
        
        peakfreqsg, peakampsg = segregatepeak(peakfreq+fnmin,peakamp,contmin,ampthr)
        
        freqmat[idx,:] = peakfreqsg
        ampmat[idx,:] = peakampsg

        if np.sum(np.logical_not(np.isnan(peakampsg)))> 0:
            peakampsg2 = copy.deepcopy(peakampsg)
            peakampsg2[np.isnan(peakampsg2)] = -np.inf
            mvC = np.max(peakampsg2, axis=1)
            miC = np.argmax(peakampsg2, axis=1)
            miR = np.argmax(mvC)
            maxampidx[n] = miR+idx[0]
            maxampfreq[n] = peakfreqsg[miR,miC[miR]]
            if np.isnan(maxampfreq[n]) == False:
                maxampval[n] = mtsp[round(maxampfreq[n]),maxampidx[n]]

        meanfreq[n] = np.nanmean((freqmat[idx,0])/fftsize*fs)
        ft = (peakfreqsg[:,0])/fftsize*fs
        cvfreq[n] = np.nanstd(ft,axis=0)/meanfreq[n]

    freqtrace = freqmat/fftsize*fs
    amptrace = ampmat
    maxfreq = maxampfreq/fftsize*fs

    return freqtrace, amptrace, maxampval, maxampidx, maxfreq, meanfreq, cvfreq

def procfun(params, wav,fs,fftsize,med,thresh):
    
    timestep = params['timestep']
    gapmin = params['gapmin']
    durmin = params['durmin']
    durmax = params['durmax']
    margin =  params['margin']
    freqmin = params['freqmin']
    freqmax = params['freqmax']
    threshval = params['threshval']
    bandwidth = params['bandwidth']
    liftercutoff = params['liftercutoff']
    
    mtsp = multitaperspec(wav,fs,fftsize,timestep)

    [fltnd,med] = flattening(mtsp,med,liftercutoff)
    
    thresh = estimatethresh(fltnd,fs,freqmin,freqmax,threshval)

    thrshd = thresholding(fltnd,fs,freqmin,freqmax,thresh)

    onoffset, onoffsetm, onoffsig, contflg = detectonoffset(thrshd,fs,timestep,gapmin,durmin,durmax,margin)

    freqtrace, amptrace, maxampval, maxampidx, maxfreq, meanfreq, cvfreq = specpeaktracking(mtsp,fltnd,fs,timestep,freqmin,freqmax,onoffset,margin,bandwidth)

    return mtsp, fltnd, onoffset, onoffsetm, freqtrace, amptrace, maxampval, maxampidx, maxfreq, meanfreq, cvfreq, thresh, med, contflg

def soundsynthesis(freq,amp,tvec,fs,rfs,freqmap):

    # time vector
    rt = np.arange(1, round(rfs*np.max(tvec)), 1)/rfs
    # process frequency
    nid = np.logical_not(np.isnan(freq))
    npf = freq[nid]
    npf = np.concatenate([np.array([np.mean(npf)]), npf, np.array([np.mean(npf)])], axis=0)
    nT = np.concatenate([np.array([1/fs]), tvec[nid], np.array([np.max(tvec)])], axis=0)
    fil = scipy.interpolate.interp1d(nT, npf, fill_value='extrapolate')
    p = fil(rt)
    pm = p/(fs/2)*(freqmap[1]-freqmap[0])+freqmap[0]
    pm[pm<100] = 100
    # process amplitude
    a2 = copy.deepcopy(amp)
    a2[np.isnan(a2)] = -120
    fil = scipy.interpolate.interp1d(tvec, a2, fill_value='extrapolate')
    a3 = fil(rt)
    a4 = 10**(a3/20)
    afil = scipy.signal.lfilter(np.ones(128)/128, 1, np.concatenate([a4, np.zeros(64)]), axis=0)
    afil = afil[64:]
    ampli = 0.2 * afil/np.max(afil)
    ampli[20*np.log10(afil)<-120] = 0
    # synthesis
    omega = 2*np.pi*pm
    ph = np.cumsum(omega/rfs)
    sig = np.sin(ph)
    snd = sig*ampli + 2**-16*np.random.randn(sig.shape[0])

    return snd

def getsubsegment(fltnd, peakfreqsg, peakampsg):
    
    img = np.zeros(fltnd.shape, np.uint8)
    for i in range(4):
        f = peakfreqsg[:,i]
        t = np.arange(0,f.shape[0])
        I = np.logical_not(np.isnan(f))
        t = np.floor(t[I])
        f = np.floor(f[I])
        img[f.astype(int), t.astype(int)] = 1

    se = np.ones([3, 3], np.uint8)
    img = cv2.dilate(img, se)
    img = cv2.dilate(img, se)
    img = cv2.erode(img, se)
    
    C = cv2.cornerHarris(img, 2, 3, 0.2)    ##### TODO: adjust parameters
    I = np.argwhere(C>0.01*C.max())
    
    img_ori = copy.deepcopy(img)

    for i in range(I.shape[0]):
        img[max(0,I[i,0]-7):min(img.shape[0],I[i,0]+7), max(0,I[i,1]-1):min(img.shape[1],I[i,1]+1)] = 0

    ret, markers = cv2.connectedComponents(img)

    markers = cv2.watershed(np.zeros([img.shape[0],img.shape[1], 3], np.uint8), markers)

    P = []
    for i in range(4):
        f = peakfreqsg[:,i]
        a = peakampsg[:,i]
        t = np.arange(0,f.shape[0])
        I = np.logical_not(np.isnan(f))
        p = np.array([t[I], f[I], a[I]]).T
        P.append(p)
    P = np.concatenate(P, axis=0)

    A = np.zeros([P.shape[0],1])
    for i in range(P.shape[0]):
        A[i] = markers[int(P[i,1]), int(P[i,0])]
    
    P = np.concatenate([P, A], axis=1)
    P = P[P[:,3]>0, :]

    if P.shape[0] > 0:
        for i in range(1, int(np.max(P[:,3])+1)):
            I = P[:,3] == i
            if np.sum(I) < 7:
                P[I,3] = -1

        I = np.argsort(P[:, 3])
        ss = P[I,:]
    else:
        ss = None

    return ss

def segfun(startid,outp,prefix,inputimg,imrng,wav,fs,timestep,margin,onoffset,tvec,amptrace,freqtrace,wavflg,imgflg,trcflg,usvcamflg):
    fftsize = (inputimg.shape[0]-1)*2
    step = round(timestep*fs)
    onset = onoffset[:,0]
    offset = onoffset[:,1]

    """im = np.flipud(((inputimg-imrng[0])/(imrng[1]-imrng[0])*64).astype(np.uint8))
    cv2.imwrite('out.png',im)
    print(im.shape)"""

    im = (1 - np.flipud((inputimg-imrng[0])/(imrng[1]-imrng[0])))*255
    im[im<0] = 0
    im[im>255] = 255

    for n in range(onset.shape[0]):
        rng = np.array([max(round((onset[n]-margin)*fs),0), min(round((offset[n]+margin)*fs), wav.shape[0])])
        rngs = np.round((rng-fftsize/2)/step)
        rng2 = np.array([max(rngs[0]-1,0), min(rngs[1],im.shape[1])], dtype=int)

        # wave write
        if wavflg == 1: 
            fname = outp + '/{:s}_{:04d}.wav'.format(prefix, n+startid)
            scipy.io.wavfile.write(fname, rate=fs, data=wav[(rng[0]-1):rng[1]])

        # jpg write
        if imgflg == 1:
            imseg = im[:,rng2[0]:rng2[1]]
            fname = outp + '/{:s}_{:04d}.jpg'.format(prefix, n+startid)
            _, buf = cv2.imencode('*.jpg', imseg)
            buf.tofile(fname)

        # trace write
        if trcflg == 1:
            af = np.concatenate([amptrace[:,0].reshape([-1,1]), freqtrace[:,0].reshape([-1,1]),
                                 amptrace[:,1].reshape([-1,1]), freqtrace[:,1].reshape([-1,1]),
                                 amptrace[:,2].reshape([-1,1]), freqtrace[:,2].reshape([-1,1])], axis=1)
            dat = np.concatenate([tvec[rng2[0]:rng2[1]].reshape([-1,1]), af[rng2[0]:rng2[1],:]], axis=1)
            # CSV file
            if usvcamflg:
                fname = outp + '/{:s}_{:04d}.ori.csv'.format(prefix, n+startid)
            else:
                fname = outp + '/{:s}_{:04d}.csv'.format(prefix, n+startid)
            np.savetxt(fname, dat, delimiter=',', fmt='%.10e')


        if usvcamflg:
            # get & output subsegments  %%%% added by JM, 2021/4/12
            if trcflg == 1:
                ft = freqtrace[rng2[0]:rng2[1],:]*fftsize/fs + 1
                at = amptrace[rng2[0]:rng2[1],:]
                t = tvec[rng2[0]:rng2[1]]

                imseg = im[:,rng2[0]:rng2[1]]
                ss = getsubsegment(imseg, ft, at)

                if ss is None:
                    ss = np.empty([0,4])
                else:
                    ss[:,0] = t[ss[:,0].astype(int)]
                    ss[:,1] = (ss[:,1]-1)*fs/fftsize

                    # mark subsegment totally within the margins 
                    for i in range(1, int(np.max(ss[:,3])+1)):
                        I = ss[:,3] == i
                        if np.sum(I) == 0:
                            continue
                        if (np.max(ss[I,0]) < np.min(t)+margin) or (np.min(ss[I,0]) > np.max(t)-margin):
                            ss[I,3] = -2
                    I = np.logical_or(ss[:,0] < min(t)+margin, ss[:,0] > max(t)-margin)
                    I = np.logical_and(I, ss[:,3] == -1)
                    
                    # exclude very weak amplitude segments
                    for i in range(1, int(np.max(ss[:,3])+1)):
                        I = ss[:,3] == i
                        if np.sum(I) == 0:
                            continue
                        if np.median(ss[I,2]) < 3:  # 3dB from baseline
                            ss[I,3] = -3

                fname = outp + '/{:s}_{:04d}.ss.csv'.format(prefix, n+startid)
                np.savetxt(fname, ss, delimiter=',', fmt='%.10e')

def wavread(wf, pos, npoints):
    wf.setpos(pos)
    buf = wf.readframes(npoints)
    if wf.getsampwidth() == 2:
        x = np.frombuffer(buf, dtype='int16') / np.iinfo(np.int16).max
    elif wf.getsampwidth() == 4:
        x = np.frombuffer(buf, dtype='int32') / np.iinfo(np.int32).max

    return x

def proc_wavfile(params, fp, savefp, outp, fname_audiblewav=None, ui_thread=None, usvcamflg=False):

    prefix = os.path.splitext(os.path.basename(fp))[0]

    timestep = params['timestep']
    margin =  params['margin']
    durmax = params['durmax']
    wavflg = params['wavfileoutput']
    imgflg = params['imageoutput']
    trcflg = params['traceoutput']
    readsize = params['readsize']
    fftsize = params['fftsize']
    freqmin = params['freqmin']
    freqmax = params['freqmax']
    threshval = params['threshval']
    durmin = params['durmin']
    gapmin = params['gapmin']
    fltflg = params['imagetype']
    mapL = params['mapL']
    mapH = params['mapH']

    with wave.open(fp, mode='rb') as wf:

        wavsize = wf.getnframes()
        fs = wf.getframerate()
        nreadsize= round(readsize*fs)
        nread = math.ceil(wavsize/nreadsize)
        fvec = np.arange(0, fs/2+1, fs/fftsize, dtype=float)
        step = round(timestep*fs)

        # CSV setting
        with open(savefp, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['#','start','end','duration','maxfreq','maxamp','meanfreq','cvfreq'])
        
        # segmentation setting
        os.makedirs(outp, exist_ok=True)

        prevn = 0
        prevlast = 0
        med = None
        thresh = []

        for r in tqdm(range(1,nread+1)):
            
            # read
            rng = [prevlast, min(r*nreadsize,wavsize)]
            if (rng[1]-rng[0])<fftsize*2:
                break
            wav = wavread(wf, rng[0], rng[1]-rng[0])

            # process
            mtsp, fltnd, onoffset, onoffsetm, freqtrace, amptrace, maxampval, maxampidx, maxfreq, meanfreq, cvfreq, thresh, med, contflg = procfun(params, wav,fs,fftsize,med,thresh)

            dur = np.diff(onoffset, axis=1)
            ronoffset = onoffset+(prevlast+1)/fs
            ronoffsetm = onoffsetm+(prevlast+1)/fs
            nstep = mtsp.shape[1]
            tvec = (np.arange(0,nstep,1)*step+fftsize/2)/fs
            rtvec = tvec+(prevlast+1)/fs

            mtsimrng = np.nanmedian(mtsp) + np.array([0, 40])
            fltndimrng = np.array([0, 30])

            # draw MTSP, flattened spec, traces in GUI
            if ui_thread is not None:

                def cvt_specimg_for_disp(inputimg, imrng):
                    im = (inputimg-imrng[0])/(imrng[1]-imrng[0])
                    im[im<0.0] = 0.0
                    im[im>1.0] = 1.0
                    return im
                
                mtsp_disp = cvt_specimg_for_disp(mtsp, mtsimrng)
                fltnd_disp = cvt_specimg_for_disp(fltnd, fltndimrng)
                x_disp, y_disp = np.meshgrid(tvec,fvec/1000)

                if onoffset.shape[0] > 0:
                    nonsylzone = np.zeros([onoffsetm.shape[0]+1,2])
                    nonsylzone[1:,0] = onoffsetm[:, 1]
                    nonsylzone[:-1,1] = onoffsetm[:, 0]
                    nonsylzone[-1, 1] = np.max(tvec)
                    
                    margzone = np.zeros([onoffset.shape[0]*2, 2])
                    margzone[:onoffset.shape[0],0] = onoffset[:,0]-margin
                    margzone[:onoffset.shape[0],1] = onoffset[:,0]
                    margzone[onoffset.shape[0]:,0] = onoffset[:,1]
                    margzone[onoffset.shape[0]:,1] = onoffset[:,1]+margin
                    margzone = margzone[np.argsort(margzone[:, 0]),:]

                    idx = np.argwhere((margzone[1:,0]-margzone[:-1,1])>0)
                    idx = idx.ravel()
                    ons = np.concatenate([np.array([margzone[0,0]]), margzone[idx+1,0]], axis=0)
                    offs = np.concatenate([margzone[idx,1], np.array([margzone[-1,1]])], axis=0)
                    margzone = np.concatenate([ons[:,np.newaxis], offs[:,np.newaxis]], axis=1)
                else:
                    nonsylzone = np.array([[0, np.max(tvec)]])
                    margzone = np.array([[0, 0]])

                info_str = 'File: ' + fp + '...  ({:d}/{:d} blocks)'.format(r, nread)
                figdata = FigData(x_disp, y_disp, mtsp_disp, fltnd_disp, nonsylzone, margzone, tvec, fvec, freqtrace, fs, freqmin, freqmax, info_str)
                
                if ui_thread.thread().isInterruptionRequested():
                    return
                
                ui_thread.progress.emit(figdata)

            # save csv
            with open(savefp, 'a', newline='') as f:
                writer = csv.writer(f)
                if ronoffset.shape[0] > 0:
                    for n in range(ronoffset.shape[0]):
                        writer.writerow([n+prevn+1,ronoffset[n,0],ronoffset[n,1],dur[n,0]*1000,maxfreq[n]/1000,maxampval[n],meanfreq[n]/1000,cvfreq[n]])
                
            # segmentation
            if fltflg==1:
                inputimg = fltnd
                imrng = fltndimrng
            else:
                inputimg = mtsp
                imrng = mtsimrng

            segfun(prevn+1,outp,prefix,inputimg,imrng,wav,fs,timestep,margin,onoffset,rtvec,amptrace,freqtrace,wavflg,imgflg,trcflg, usvcamflg)
            

            # calc
            prevn = onoffset.shape[0]+prevn
            if contflg==1:
                if onoffset.shape[0] == 0:
                    prevlast = rng[1] - durmax * fs
                else:
                    prevlast = round(onoffset[-1,1]*fs)+prevlast
                    
            else:
                prevlast = rng[1]

    # output synth sound file
    if fname_audiblewav is not None:
        playfs = 44100

        if usvcamflg:
            L = glob.glob(outp + '/*.ori.csv')
        else:
            L = glob.glob(outp + '/*.csv')
        data = np.zeros([0,3])
        for l in L:
            d = np.loadtxt(l, delimiter=',')
            data = np.concatenate([data, d[:,0:3]], axis=0)

        if data.shape[0] == 0:
            synthsnd = np.zeros([playfs,1])
        else:
            data = data[np.argsort(data[:, 0])]
            _, I = np.unique(data[:,0], return_index=True)
            data = data[I,:]
            synthsnd = soundsynthesis(data[:,2],data[:,1],data[:,0],fs,playfs,[mapL, mapH])

        scipy.io.wavfile.write(fname_audiblewav, rate=playfs, data=(synthsnd*32765).astype(np.int16))


