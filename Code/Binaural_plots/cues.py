import os
import warnings
import soundfile as sf
import numpy as np
from scipy.signal import hilbert, firwin2, lfilter, fftconvolve
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

def calc_pdf_cues(audio_file):
    # Get the current file path
    audioin = audio_file

    # Define audio parameters
    offset = 0                     # No offset
    sfreq = 16000                  # Sampling frequency
    numsamples = 20 * sfreq         # Number of samples

    # Define auditory periphery parameters
    fcenter = 500                  # Center frequency [Hz]
    SPL = 60                       # Sound pressure level [dB]
    maxitd = 1.0                   # ITD range [ms]
    maxild = 7                     # ILD range [dB]
    tau = [10, 10, 10]             # Time constants [ms]
    statoffset = 400                # Statistics offset [ms]

    # Define cue selection parameters
    c0 = 0.98                # Cue selection threshold

    res = 30               # Resolution for PDF computation

    # --------------------------------------------------------
    # Process the audio file
    # --------------------------------------------------------

    # Check if the file exists
    if not os.path.isfile(audioin):
        warnings.warn(f"File does not exist: {audioin}. Skipping...")
        return None

    # Read audio file info
    info = sf.info(audioin)
    Nmax = info.frames

    # Determine number of samples
    if numsamples == 0:
        N = Nmax
    else:
        N = numsamples
    if N + offset > Nmax:
        N = Nmax - offset

    start_frame = offset
    stop_frame = offset + N

    # Read the audio data
    s, sfreq = sf.read(audioin, start=start_frame, stop=stop_frame)
    # s = s/np.max(abs(s))
    # Ensure stereo input
    if s.ndim < 2 or s.shape[1] < 2:
        warnings.warn(f"File is not stereo: {audioin}. Skipping...")
        return None

    # Separate channels (transpose to match MATLAB's row vectors)
    in1 = s[:, 0].T
    in2 = s[:, 1].T
    del s

    # Apply perceptual model to compute binaural cues
    statoffsetN = statoffset * sfreq / 1000.0
    ild, itd, ic, pow_ = ild_itd_ic(sfreq, fcenter, in1, in2, maxitd, maxild, tau, statoffsetN, SPL)
    # Compute PDF with cue selection
    wild_itd_pdf, wild_pdf, witd_pdf, ildaxis, itdaxis = pdf_c0(
        ild, itd, ic, pow_, maxild, maxitd, res, c0
    )
    return wild_itd_pdf, wild_pdf, witd_pdf, ildaxis, itdaxis

import numpy as np

def ild_itd_ic(sfreq, fcenter, in1, in2, maxitd, maxild, tau, ofst, spl):
    """
    Compute binaural cues (ILD, ITD, IC, power) for one pair of critical bands.
    Fully translated from MATLAB (C. Faller & J. Merimaa 07-2003).
    """

    # --- Parameters ---
    use_auditory_filter = True
    add_noise = False

    # Ensure row vectors
    in1 = np.ravel(in1)
    in2 = np.ravel(in2)
    N = max(len(in1), len(in2))

    # Tau handling
    if np.isscalar(tau):
        tau_ic = tau
        tau_ild = tau
        tau_itd = tau
    else:
        tau_ild, tau_itd, tau_ic = tau

    maxitds = int(round(maxitd * sfreq / 1000.0))
    alpha = 1.0 / (tau_ic * sfreq / 1000.0)
    alpha2 = 1.0 / (tau_ild * sfreq / 1000.0)
    alpha3 = 1.0 / (tau_itd * sfreq / 1000.0)

    # --- Auditory filterbank and transduction ---
    if use_auditory_filter:
        EarQ = 9.26449
        minBW = 24.7
        fcoefs = mmakeerbfilters(sfreq, np.array([fcenter]), None, EarQ, minBW)

        # Apply auditory filterbank
        in1 = merbfilterbank(in1, fcoefs)[0]
        in2 = merbfilterbank(in2, fcoefs)[0]

        # Add noise based on threshold curves
        if add_noise:
            hthr = np.array([
                [125, 47.9],
                [250, 28.3],
                [500, 12.6],
                [1000, 6.8],
                [1500, 6.7],
                [2000, 7.8],
                [3000, 7.6],
                [4000, 8.7],
                [6000, 11.9],
                [8000, 11.6]
            ])
            thr = hthr[np.argmin(np.abs(hthr[:, 0] - fcenter)), 1]

            ns1 = merbfilterbank(np.random.randn(len(in1)), fcoefs)[0]
            ns2 = merbfilterbank(np.random.randn(len(in2)), fcoefs)[0]

            idx1 = np.where(np.abs(in1) > 0)[0]
            idx2 = np.where(np.abs(in2) > 0)[0]
            offset1 = idx1[0] if idx1.size > 0 else 0
            offset2 = idx2[0] if idx2.size > 0 else 0

            power1 = np.sum(in1[offset1:] ** 2)
            power2 = np.sum(in2[offset2:] ** 2)
            pwr = (power1 + power2) / 2.0
            offset = int(round((offset1 + offset2) / 2))

            gain1 = np.sqrt(pwr / np.sum(ns1[offset:] ** 2)) * 10 ** ((thr - spl) / 20)
            gain2 = np.sqrt(pwr / np.sum(ns2[offset:] ** 2)) * 10 ** ((thr - spl) / 20)
            in1 += gain1 * ns1
            in2 += gain2 * ns2

        # Neural transduction
        in1, _, _ = mmonauraltransduction(in1[np.newaxis, :], 'envelope', sfreq, 0)
        in2, _, _ = mmonauraltransduction(in2[np.newaxis, :], 'envelope', sfreq, 0)
        in1 = np.where(in1.flatten() > 0, in1.flatten(), 0)
        in2 = np.where(in2.flatten() > 0, in2.flatten(), 0)

    # --- Binaural cue computation (Matlab loop mode) ---
    ild, itd, ic, pow_ = [], [], [], []

    num = np.ones(2 * maxitds + 1) * 1e-40
    den1 = num.copy()
    den2 = num.copy()
    num_ = num.copy()
    den1_ = den1.copy()
    den2_ = den2.copy()
    lev1 = num.copy()
    lev2 = num.copy()

    
    for i in range(maxitds, N - maxitds):
        # Short analysis window centered at current sample
        S1 = in1[i - maxitds : i + maxitds + 1]          # (2*maxitds + 1,)
        S2 = in2[i - maxitds : i + maxitds + 1]          # same length

        # Reverse S2 for cross-correlation style alignment
        S2_rev = np.flip(S2)

        # --- Interaural Coherence (IC) ---
        num  = alpha * (S1 * S2_rev) + (1 - alpha) * num
        den1 = alpha * (S1**2) + (1 - alpha) * den1
        den2 = alpha * (S2_rev**2) + (1 - alpha) * den2
        ic_func = num / np.sqrt(den1 * den2 + 1e-20)

        max_idx = np.argmax(ic_func)
        ic.append(ic_func[max_idx])

        # --- ITD ---
        num_  = alpha3 * (S1 * S2_rev) + (1 - alpha3) * num_
        den1_ = alpha3 * (S1**2) + (1 - alpha3) * den1_
        den2_ = alpha3 * (S2_rev**2) + (1 - alpha3) * den2_
        # 0-lag is at index = maxitds
        itd.append((max_idx - maxitds) / sfreq * 1000.0)

        # --- ILD ---
        lev1 = alpha2 * (S1**2) + (1 - alpha2) * lev1
        lev2 = alpha2 * (S2_rev**2) + (1 - alpha2) * lev2
        ild_val = 10 * np.log10((lev1[max_idx] + 1e-40) / (lev2[max_idx] + 1e-40))
        ild_val = np.clip(ild_val, -maxild, maxild)
        ild.append(ild_val)

        # --- Power ---
        pow_.append(lev1[max_idx] + lev2[max_idx])

    # --- Apply offset ---
    idx = slice(int(ofst), None)
    ild = np.array(ild)[idx]
    itd = np.array(itd)[idx]
    ic = np.array(ic)[idx]
    pow_ = np.array(pow_)[idx]

    return ild, itd, ic, pow_


def pdf_c0(ild, itd, ic, pow_, maxild, maxitd, res, c0, debug=False):
    """
    Compute ILD-ITD probability density functions.
    """
    # Axes edges
    axis_ild = np.linspace(-maxild, maxild, 2 * res + 1)
    axis_itd = np.linspace(-maxitd, maxitd, 2 * res + 1)

    # Mask selection
    pw0 = 0.2 * np.mean(pow_)
    mask = (ic > c0)
    if np.sum(mask) == 0:
        if debug:
            print("[WARN] No cues above c0, falling back to power filter")
        mask = pow_ > pw0

    ild_sel = ild[mask]
    itd_sel = itd[mask]

    if ild_sel.size == 0 or itd_sel.size == 0:
        if debug:
            print("[WARN] No cues pass mask, returning empty PDFs")
        M = len(axis_ild) - 1
        return np.zeros((M, M)), np.zeros(M), np.zeros(M), axis_ild[:-1], axis_itd[:-1]

    # Joint histogram (2D)
    ild_itd_pdf, _, _ = np.histogram2d(
        ild_sel, itd_sel, bins=[axis_ild, axis_itd]
    )

    # Marginals (1D)
    ild_pdf, _ = np.histogram(ild_sel, bins=axis_ild)
    itd_pdf, _ = np.histogram(itd_sel, bins=axis_itd)

    # Normalize
    if np.max(ild_itd_pdf) > 0:
        ild_itd_pdf = ild_itd_pdf.astype(float) / np.max(ild_itd_pdf)
    ild_pdf = ild_pdf.astype(float)
    itd_pdf = itd_pdf.astype(float)
    if np.max(ild_pdf) > 0:
        ild_pdf /= np.max(ild_pdf)
    if np.max(itd_pdf) > 0:
        itd_pdf /= np.max(itd_pdf)

    # Bin centers for axes
    axis_ild_centers = (axis_ild[:-1] + axis_ild[1:]) / 2
    axis_itd_centers = (axis_itd[:-1] + axis_itd[1:]) / 2

    if debug:
        print(f"[DEBUG] wild_itd_pdf max: {np.max(ild_itd_pdf)}")
        print(f"[DEBUG] wild_pdf max: {np.max(ild_pdf)}")
        print(f"[DEBUG] witd_pdf max: {np.max(itd_pdf)}")

    return ild_itd_pdf[::-1,:], ild_pdf, itd_pdf, axis_ild_centers, axis_itd_centers



def mmakeerbfilters(fs, cfarray, lowFreq=None, EarQ=9.26449, minBW=24.7):
    """
    Computes filter coefficients for a gammatone filterbank (ERB filters).
    Equivalent to MakeERBFilters.m from Slaney's toolbox.

    Parameters
    ----------
    fs : float
        Sampling frequency.
    cfarray : array-like
        Center frequencies for each filter.
    lowFreq : ignored (only for compatibility)
    EarQ : float
        Quality factor (default 9.26449).
    minBW : float
        Minimum bandwidth (default 24.7).

    Returns
    -------
    fcoefs : ndarray
        Filter coefficients [A0, A11, A12, A13, A14, A2, B0, B1, B2, gain]
    """
    cf = np.atleast_1d(cfarray).astype(float)
    T = 1.0 / fs
    order = 1

    ERB = ((cf / EarQ) ** order + minBW ** order) ** (1.0 / order)
    B = 1.019 * 2 * np.pi * ERB

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * np.cos(2 * cf * np.pi * T) / np.exp(B * T)
    B2 = np.exp(-2 * B * T)

    A11 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) +
            2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
    A12 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) -
            2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
    A13 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) +
            2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2
    A14 = -(2 * T * np.cos(2 * cf * np.pi * T) / np.exp(B * T) -
            2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(2 * cf * np.pi * T) / np.exp(B * T)) / 2

    gain = np.abs(
        (-2 * np.exp(4j * cf * np.pi * T) * T +
         2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T *
         (np.cos(2 * cf * np.pi * T) - np.sqrt(3 - 2 ** 1.5) * np.sin(2 * cf * np.pi * T))) *
        (-2 * np.exp(4j * cf * np.pi * T) * T +
         2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T *
         (np.cos(2 * cf * np.pi * T) + np.sqrt(3 - 2 ** 1.5) * np.sin(2 * cf * np.pi * T))) *
        (-2 * np.exp(4j * cf * np.pi * T) * T +
         2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T *
         (np.cos(2 * cf * np.pi * T) - np.sqrt(3 + 2 ** 1.5) * np.sin(2 * cf * np.pi * T))) *
        (-2 * np.exp(4j * cf * np.pi * T) * T +
         2 * np.exp(-(B * T) + 2j * cf * np.pi * T) * T *
         (np.cos(2 * cf * np.pi * T) + np.sqrt(3 + 2 ** 1.5) * np.sin(2 * cf * np.pi * T))) /
        (-2 / np.exp(2 * B * T) - 2 * np.exp(4j * cf * np.pi * T) +
         2 * (1 + np.exp(4j * cf * np.pi * T)) / np.exp(B * T)) ** 4
    )

    allfilts = np.ones_like(cf)
    fcoefs = np.vstack([A0 * allfilts, A11, A12, A13, A14,
                        A2 * allfilts, B0 * allfilts, B1, B2, gain]).T
    return fcoefs


def merbfilterbank(x, fcoefs):
    """
    Applies a gammatone filterbank to a signal.

    Parameters
    ----------
    x : ndarray
        Input waveform (1D).
    fcoefs : ndarray
        Filter coefficients from mmakeerbfilters.

    Returns
    -------
    output : ndarray
        Filterbank output, shape (n_filters, len(x)).
    """
    if x.ndim > 1:
        x = x.flatten()

    if fcoefs.shape[1] != 10:
        raise ValueError("fcoefs must be (n_filters, 10) array")

    A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = fcoefs.T
    n_filters = fcoefs.shape[0]
    output = np.zeros((n_filters, len(x)))

    for chan in range(n_filters):
        y1 = lfilter([A0[chan] / gain[chan], A11[chan] / gain[chan], A2[chan] / gain[chan]],
                     [B0[chan], B1[chan], B2[chan]], x)
        y2 = lfilter([A0[chan], A12[chan], A2[chan]],
                     [B0[chan], B1[chan], B2[chan]], y1)
        y3 = lfilter([A0[chan], A13[chan], A2[chan]],
                     [B0[chan], B1[chan], B2[chan]], y2)
        y4 = lfilter([A0[chan], A14[chan], A2[chan]],
                     [B0[chan], B1[chan], B2[chan]], y3)
        output[chan, :] = y4

    return output

def mhalfwaverectify(x):
    return np.maximum(x, 0)

def mmonauraltransduction(multichanneldata, transduction, samplerate, infoflag=0):
    """
    Neural transduction for filterbank output.

    Returns:
    --------
    multichanneldata2 : ndarray
        Transduced data
    output_powervector : ndarray
        Power per channel
    output_maxvector : ndarray
        Max per channel
    """
    if transduction == "linear":
        multichanneldata2 = multichanneldata

    elif transduction == "hw":
        multichanneldata2 = mhalfwaverectify(multichanneldata)

    elif transduction == "log":
        multichanneldata2 = mhalfwaverectify(multichanneldata)
        multichanneldata2[multichanneldata2 < 1] = 1
        multichanneldata2 = 20 * np.log10(multichanneldata2)

    elif transduction == "power":
        multichanneldata2 = mhalfwaverectify(multichanneldata) ** 0.4

    elif transduction == "v=3":
        multichanneldata2 = mhalfwaverectify(multichanneldata) ** 3

    elif transduction == "envelope":
        compress1 = 0.23
        compress2 = 2.0
        cutoff = 425
        order = 4

        # Design lowpass filter
        lpf_freq = np.linspace(0, samplerate / 2, 10000)
        f0 = cutoff * (1.0 / (2 ** (1 / order) - 1) ** 0.5)
        lpmag = 1.0 / (1 + (lpf_freq / f0) ** 2) ** (order / 2)
        lpf = lpf_freq / (samplerate / 2)
        coeffs = firwin2(257, lpf, lpmag)

        nfilters = multichanneldata.shape[0]
        multichanneldata2 = np.zeros_like(multichanneldata)

        for f in range(nfilters):
            envelope = np.abs(hilbert(multichanneldata[f, :]))
            compressedenvelope = (envelope ** (compress1 - 1)) * multichanneldata[f, :]
            rectifiedenvelope = np.maximum(compressedenvelope, 0)
            rectifiedenvelope = rectifiedenvelope ** compress2
            multichanneldata2[f, :] = fftconvolve(rectifiedenvelope, coeffs, mode="same")

    else:
        raise ValueError(f"Unknown transduction type: {transduction}")

    output_powervector = np.sqrt(np.mean(multichanneldata2 ** 2, axis=1))
    output_maxvector = np.max(multichanneldata2, axis=1)
    return multichanneldata2, output_powervector, output_maxvector

def plot_pdf(ild_itd_pdf, ild_pdf, itd_pdf, axis_ild, axis_itd, 
             ild_ff=None, itd_ff=None, ildrange=None, itdrange=None, maxval=1,color='gray'):
    """
    Plot ILD-ITD joint PDF, ILD marginal, and ITD marginal with corrected alignment and more ticks.
    """
    # Defaults
    ild_ff = [] if ild_ff is None else ild_ff
    itd_ff = [] if itd_ff is None else itd_ff
    ildrange = np.max(np.abs(axis_ild)) if ildrange is None else ildrange
    itdrange = np.max(np.abs(axis_itd)) if itdrange is None else itdrange
    ildrange =  np.round(ildrange).astype(float) 
    itdrange =  np.round(itdrange).astype(float) 

    n_ticks = 11
 
    box_ild_itd = [0.27, 0.27, 0.6, 0.6]
    box_itd     = [0.27, 0.1, 0.6, 0.15]
    box_ild     = [0.1,  0.27, 0.15, 0.6]

    clim = (0, maxval)
    g = []
    cmaps = {'gray':'gray_r','red':'Reds','green':'Greens','blue':'Blues'}
    if color not in cmaps.keys():
        cmap = 'gray_r'
    else:
        cmap = cmaps[color]
    if cmap == 'gray_r':
        color='k'
        
    # --- Joint PDF ---
    ax1 = plt.axes(box_ild_itd)
    im = ax1.imshow(
        ild_itd_pdf,
        extent=[axis_itd[0], axis_itd[-1], axis_ild[0], axis_ild[-1]],
        origin='lower',
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        aspect='auto'
    )
    ax1.set_xlim(-itdrange, itdrange)
    ax1.set_ylim(-ildrange, ildrange)

    # Add more ticks (5 ticks for each axis)
    ax1.set_xticks(np.linspace(-itdrange, itdrange, n_ticks))
    ax1.set_yticks(np.linspace(-ildrange, ildrange, n_ticks))

    # Hide labels (they will align with the marginals)
    ax1.tick_params(labelbottom=False, labelleft=False)
    g.append(ax1)

    # --- ITD PDF ---
    ax2 = plt.axes(box_itd, sharex=ax1)
    ax2.plot(axis_itd, itd_pdf, color, linewidth=1.1)
    for t in itd_ff:
        ax2.axvline(x=t, linestyle=':', color='k', linewidth=1.3)
    ax2.set_xlim(-itdrange, itdrange)
    ax2.set_ylim(0, 1)

    # match ticks with middle box
    ax2.set_xticks(np.linspace(-itdrange, itdrange, n_ticks))
    ax2.set_xlabel('ITD [ms]')
    ax2.set_ylabel('P(ITD)')
    ax2.tick_params(axis='y', labelleft=False)
    g.append(ax2)

    # --- ILD PDF ---
    ax3 = plt.axes(box_ild, sharey=ax1)
    ax3.plot(ild_pdf, axis_ild[::-1], color, linewidth=1.1)
    for l in ild_ff:
        ax3.axhline(y=-l, linestyle=':', color='k', linewidth=1.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-ildrange, ildrange)

    # match ticks with middle box
    ax3.set_yticks(np.linspace(-ildrange, ildrange, n_ticks))

    ax3.set_xlabel('P(ILD)')
    ax3.set_ylabel('ILD [dB]')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.xaxis.set_label_position('top')
    g.append(ax3)

    # Styling
    for ax in g:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_color([0.5, 0.5, 0.5])
        ax.tick_params(width=1.5)
        ax.title.set_weight('bold')

    return g

def plot_pdf2(ild_itd_pdf1, ild_pdf1, itd_pdf1,ild_itd_pdf2, ild_pdf2, itd_pdf2, axis_ild, axis_itd, 
             ild_ff=None, itd_ff=None, ildrange=None, itdrange=None, maxval=1,color='gray'):
    """
    Plot ILD-ITD joint PDF, ILD marginal, and ITD marginal with corrected alignment and more ticks.
    """
    # Defaults
    ild_ff = [] if ild_ff is None else ild_ff
    itd_ff = [] if itd_ff is None else itd_ff
    ildrange = np.max(np.abs(axis_ild)) if ildrange is None else ildrange
    itdrange = np.max(np.abs(axis_itd)) if itdrange is None else itdrange
    ildrange =  np.round(ildrange).astype(float) 
    itdrange =  np.round(itdrange).astype(float) 

    n_ticks = 11
 
    box_ild_itd = [0.27, 0.27, 0.6, 0.6]
    box_itd     = [0.27, 0.1, 0.6, 0.15]
    box_ild     = [0.1,  0.27, 0.15, 0.6]

    clim = (0, maxval)
    g = []

    cmap_reds = plt.cm.get_cmap('Reds')(np.linspace(0, 1, 256))    
    cmap_reds[:,-1] = 0.75
    cmap_reds[0,-1]=0.0
    cmap_reds = ListedColormap(cmap_reds,'Reds_transparent_zero')

    cmap_blues = plt.cm.get_cmap('Blues')(np.linspace(0, 1, 256))    
    cmap_blues[:,-1] = 0.75
    cmap_blues[0,-1]=0.0
    cmap_blues = ListedColormap(cmap_blues,'Blues_transparent_zero')

    # --- Joint PDF ---
    ax1 = plt.axes(box_ild_itd)
    im = ax1.imshow(
        ild_itd_pdf1,
        extent=[axis_itd[0], axis_itd[-1], axis_ild[0], axis_ild[-1]],
        origin='lower',
        cmap=cmap_reds,
        vmin=clim[0],
        vmax=clim[1],
        aspect='auto'
    )
    im = ax1.imshow(
        ild_itd_pdf2,
        extent=[axis_itd[0], axis_itd[-1], axis_ild[0], axis_ild[-1]],
        origin='lower',
        cmap=cmap_blues,
        vmin=clim[0],
        vmax=clim[1],
        aspect='auto'
    )
    
    ax1.set_xlim(-itdrange, itdrange)
    ax1.set_ylim(-ildrange, ildrange)

    # Add more ticks (5 ticks for each axis)
    ax1.set_xticks(np.linspace(-itdrange, itdrange, n_ticks))
    ax1.set_yticks(np.linspace(-ildrange, ildrange, n_ticks))

    # Hide labels (they will align with the marginals)
    ax1.tick_params(labelbottom=False, labelleft=False)
    g.append(ax1)

    # --- ITD PDF ---
    ax2 = plt.axes(box_itd, sharex=ax1)
    ax2.plot(axis_itd, itd_pdf1, 'red', linewidth=1.1)
    ax2.plot(axis_itd, itd_pdf2, 'blue', linewidth=1.1)
    for t in itd_ff:
        ax2.axvline(x=t, linestyle=':', color='k', linewidth=1.3)
    ax2.set_xlim(-itdrange, itdrange)
    ax2.set_ylim(0, 1)

    # match ticks with middle box
    ax2.set_xticks(np.linspace(-itdrange, itdrange, n_ticks))
    ax2.set_xlabel('ITD [ms]')
    ax2.set_ylabel('P(ITD)')
    ax2.tick_params(axis='y', labelleft=False)
    g.append(ax2)

    # --- ILD PDF ---
    ax3 = plt.axes(box_ild, sharey=ax1)
    ax3.plot(ild_pdf1, axis_ild[::-1], 'red', linewidth=1.1)
    ax3.plot(ild_pdf2, axis_ild[::-1], 'blue', linewidth=1.1)

    for l in ild_ff:
        ax3.axhline(y=-l, linestyle=':', color='k', linewidth=1.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-ildrange, ildrange)

    # match ticks with middle box
    ax3.set_yticks(np.linspace(-ildrange, ildrange, n_ticks))

    ax3.set_xlabel('P(ILD)')
    ax3.set_ylabel('ILD [dB]')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.xaxis.set_label_position('top')
    g.append(ax3)

    # Styling
    for ax in g:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_color([0.5, 0.5, 0.5])
        ax.tick_params(width=1.5)
        ax.title.set_weight('bold')

    return g

def plot_pdf3(ild_itd_pdf1, ild_pdf1, itd_pdf1,ild_itd_pdf2, ild_pdf2, itd_pdf2,ild_itd_pdf3, ild_pdf3, itd_pdf3, axis_ild, axis_itd, 
             ild_ff=None, itd_ff=None, ildrange=None, itdrange=None, maxval=1,color='gray'):
    """
    Plot ILD-ITD joint PDF, ILD marginal, and ITD marginal with corrected alignment and more ticks.
    """
    # Defaults
    ild_ff = [] if ild_ff is None else ild_ff
    itd_ff = [] if itd_ff is None else itd_ff
    ildrange = np.max(np.abs(axis_ild)) if ildrange is None else ildrange
    itdrange = np.max(np.abs(axis_itd)) if itdrange is None else itdrange
    ildrange =  np.round(ildrange).astype(float) 
    itdrange =  np.round(itdrange).astype(float) 

    n_ticks = 11
 
    box_ild_itd = [0.27, 0.27, 0.6, 0.6]
    box_itd     = [0.27, 0.1, 0.6, 0.15]
    box_ild     = [0.1,  0.27, 0.15, 0.6]

    clim = (0, maxval)
    g = []

    cmap_reds = plt.cm.get_cmap('Reds')(np.linspace(0, 1, 256))    
    cmap_reds[:,-1] = 0.75
    cmap_reds[0,-1]=0.0
    cmap_reds = ListedColormap(cmap_reds,'Reds_transparent_zero')

    cmap_blues = plt.cm.get_cmap('Blues')(np.linspace(0, 1, 256))    
    cmap_blues[:,-1] = 0.75
    cmap_blues[0,-1]=0.0
    cmap_blues = ListedColormap(cmap_blues,'Blues_transparent_zero')

    cmap_grays = plt.cm.get_cmap('gray_r')(np.linspace(0, 1, 256))    
    cmap_grays[:,-1] = 0.75
    cmap_grays[0,-1]=0.0
    cmap_grays = ListedColormap(cmap_grays,'grays_transparent_zero')
    
    cmap_greens = plt.cm.get_cmap('Greens')(np.linspace(0, 1, 256))    
    cmap_greens[:,-1] = 0.75
    cmap_greens[0,-1]=0.0
    cmap_greens = ListedColormap(cmap_greens,'Greens_transparent_zero')
    # --- Joint PDF ---
    ax1 = plt.axes(box_ild_itd)
    im = ax1.imshow(
        ild_itd_pdf1,
        extent=[axis_itd[0], axis_itd[-1], axis_ild[0], axis_ild[-1]],
        origin='lower',
        cmap=cmap_greens,
        vmin=clim[0],
        vmax=clim[1],
        aspect='auto'
    )
    im = ax1.imshow(
        ild_itd_pdf2,
        extent=[axis_itd[0], axis_itd[-1], axis_ild[0], axis_ild[-1]],
        origin='lower',
        cmap=cmap_reds,
        vmin=clim[0],
        vmax=clim[1],
        aspect='auto'
    )
    im = ax1.imshow(
        ild_itd_pdf3,
        extent=[axis_itd[0], axis_itd[-1], axis_ild[0], axis_ild[-1]],
        origin='lower',
        cmap=cmap_blues,
        vmin=clim[0],
        vmax=clim[1],
        aspect='auto'
    )
    
    ax1.set_xlim(-itdrange, itdrange)
    ax1.set_ylim(-ildrange, ildrange)

    # Add more ticks (5 ticks for each axis)
    ax1.set_xticks(np.linspace(-itdrange, itdrange, n_ticks))
    ax1.set_yticks(np.linspace(-ildrange, ildrange, n_ticks))

    # Hide labels (they will align with the marginals)
    ax1.tick_params(labelbottom=False, labelleft=False)
    g.append(ax1)

    # --- ITD PDF ---
    ax2 = plt.axes(box_itd, sharex=ax1)
    ax2.plot(axis_itd, itd_pdf1, 'green', linewidth=1.1,linestyle='--')
    ax2.plot(axis_itd, itd_pdf2, 'red', linewidth=1.1,linestyle='-')
    ax2.plot(axis_itd, itd_pdf3, 'blue', linewidth=1.1,linestyle='-.')
    for t in itd_ff:
        ax2.axvline(x=t, linestyle=':', color='k', linewidth=1.3)
    ax2.set_xlim(-itdrange, itdrange)
    ax2.set_ylim(0, 1)

    # match ticks with middle box
    ax2.set_xticks(np.linspace(-itdrange, itdrange, n_ticks))
    ax2.set_xlabel('ITD [ms]')
    ax2.set_ylabel('P(ITD)')
    ax2.tick_params(axis='y', labelleft=False)
    g.append(ax2)

    # --- ILD PDF ---
    ax3 = plt.axes(box_ild, sharey=ax1)
    ax3.plot(ild_pdf1, axis_ild[::-1], 'green', linewidth=1.1,linestyle='--')
    ax3.plot(ild_pdf2, axis_ild[::-1], 'red', linewidth=1.1,linestyle='-')
    ax3.plot(ild_pdf3, axis_ild[::-1], 'blue', linewidth=1.1,linestyle='-.')

    for l in ild_ff:
        ax3.axhline(y=-l, linestyle=':', color='k', linewidth=1.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-ildrange, ildrange)

    # match ticks with middle box
    ax3.set_yticks(np.linspace(-ildrange, ildrange, n_ticks))

    ax3.set_xlabel('P(ILD)')
    ax3.set_ylabel('ILD [dB]')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.xaxis.set_label_position('top')
    g.append(ax3)

    # Styling
    for ax in g:
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_color([0.5, 0.5, 0.5])
        ax.tick_params(width=1.5)
        ax.title.set_weight('bold')

    return g


