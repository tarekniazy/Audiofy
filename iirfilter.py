import numpy as np

def iirfilter(N, Wn):
    
    Wn = np.asarray(Wn)
    if np.any(Wn <= 0) or np.any(Wn >= 1):
        raise ValueError("Digital filter critical frequencies "
                            "must be 0 < Wn < 1")

    ######## Initalize z,p,k
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1

    ###############################
    fs = 2.0
    warped = 2 * fs * np.tan(np.pi * Wn / fs)  #ANALOG FREQUENCY

    try:
        bw = float(warped[1] - warped[0])
        wo = float(np.sqrt(warped[0] * warped[1]))
    except IndexError as e:
        raise ValueError('Wn must specify start and stop frequencies for bandpass filter') from e

    z, p, k = lowpass_to_bandpass(z, p, k, wo, bw)
    z, p, k = bilinear_transformation(z, p, k, fs=fs)

    return coefficients(z, p, k)



def lowpass_to_bandpass(z, p, k, wo, bw):

    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    degree = relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = np.concatenate((z_lp + np.sqrt(z_lp**2 - wo**2),
                        z_lp - np.sqrt(z_lp**2 - wo**2)))
    p_bp = np.concatenate((p_lp + np.sqrt(p_lp**2 - wo**2),
                        p_lp - np.sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = np.append(z_bp, np.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp


def coefficients(z, p, k):
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    if len(z.shape) > 1:
        temp = np.poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * np.poly(z[i])
    else:
        b = k * np.poly(z)
    a = np.atleast_1d(np.poly(p))

    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                         np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                         np.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a

def bilinear_transformation(z, p, k, fs):

    
    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    degree = relative_degree(z, p)
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_z, p_z, k_z

def relative_degree(z, p):
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree